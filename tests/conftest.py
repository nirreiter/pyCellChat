"""
pytest plugin for R-output-based cross-validation testing.

Markers
-------
pytest.mark.r_script("r_scripts/test_foo.R")
    Module-level marker declaring the companion R script that generates
    all R outputs for this test module.

pytest.mark.ground_truth("output_name")
    Function-level marker declaring which output file this test
    compares against (``tests/ground_truths/output_name``).

CLI flag
--------
--generate-r-outputs
    Before running tests, validate R / package versions against tox.toml
    and regenerate all R output files referenced by the collected tests,
    running every R script in a single R session.
"""

from __future__ import annotations

import json
import anndata as ad
import subprocess
import textwrap
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    pass

# ── Paths ─────────────────────────────────────────────────────────────────────
TESTS_DIR = Path(__file__).parent
TOX_TOML = TESTS_DIR.parent / "tox.toml"
GROUND_TRUTH_FILEDIR = TESTS_DIR / "ground_truths"


# ══════════════════════════════════════════════════════════════════════════════
# Marker registration
# ══════════════════════════════════════════════════════════════════════════════

def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "r_script(path): path (relative to tests/) of the companion R script "
        "that generates R output files for this test module.",
    )
    config.addinivalue_line(
        "markers",
        "ground_truth(name): name of the R output JSON file (without extension) "
        "this test compares against (lives in tests/ground_truths/<name>.json).",
    )


# ══════════════════════════════════════════════════════════════════════════════
# CLI option
# ══════════════════════════════════════════════════════════════════════════════

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--generate-ground-truth",
        action="store_true",
        default=False,
        help=(
            "Regenerate ground truth files for tests."
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Generation hook: runs after collection, before tests execute
# ══════════════════════════════════════════════════════════════════════════════

def pytest_collection_finish(session: pytest.Session) -> None:
    if not session.config.getoption("--generate-ground-truth"):
        return

    # ── 1. Load r_requirements from tox.toml ──────────────────────────────────
    r_requirements = _load_r_requirements()

    # ── 2. Validate R environment ─────────────────────────────────────────────
    _validate_r_environment(r_requirements)

    # ── 3. Collect unique R scripts referenced by the gathered tests ──────────
    r_scripts: list[Path] = _collect_r_scripts(session.items)
    if not r_scripts:
        print("[generate-ground-truth] No r_script markers found; nothing to generate.")
        return

    # ── 4. Ensure output directory exists ─────────────────────────────────────
    GROUND_TRUTH_FILEDIR.mkdir(exist_ok=True)

    # ── 5. Build & run master R script (single R session) ─────────────────────
    _run_r_scripts(r_scripts)


# ══════════════════════════════════════════════════════════════════════════════
# ground_truth fixture
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def ground_truth(request: pytest.FixtureRequest):
    """
    Load the ground truth file associated with the current test via
    ``@pytest.mark.ground_truth("name")``.

    Returns the parsed JSON content (dict, list, int, float, str …).
    Skips the test if the output file does not exist yet.
    """
    marker = request.node.get_closest_marker("ground_truth")
    if marker is None:
        pytest.fail(
            "The 'ground_truth' fixture requires @pytest.mark.ground_truth('name') "
            "on the test function."
        )
    name: str = marker.args[0]
    output_path = GROUND_TRUTH_FILEDIR / name
    if not output_path.exists():
        pytest.fail(
            f"Ground truth file not found: {output_path.relative_to(TESTS_DIR.parent)}\n"
            "  Re-run with --generate-ground-truth to create it."
        )
    with open(output_path, encoding="utf-8") as fh:
        if output_path.suffix == ".json":
            return json.load(fh)
        elif output_path.suffix == ".h5ad":
            return ad.read_h5ad(output_path)
        else:
            pytest.fail(f"Invalid extension for ground truth file '{output_path.relative_to(TESTS_DIR.parent)}', only .json and .h5ad files supported!")


# ══════════════════════════════════════════════════════════════════════════════
# Internals
# ══════════════════════════════════════════════════════════════════════════════

def _load_r_requirements() -> dict:
    """
    Read [r_requirements] from tox.toml.
    Returns an empty dict (with a warning) if the section is missing
    """
    if not TOX_TOML.exists():
        print(
            f"\n[generate-ground-truth] WARNING: {TOX_TOML} not found. "
            "R version validation will be skipped."
        )
        return {}

    with open(TOX_TOML, "rb") as fh:
        config = tomllib.load(fh)

    reqs = config.get("r_requirements", {})
    if not reqs:
        print(
            "\n[generate-ground-truth] WARNING: No [r_requirements] section in "
            f"{TOX_TOML}. R version validation will be skipped."
        )
    return reqs


def _validate_r_environment(r_requirements: dict) -> None:
    """
    Run a small inline R script that checks the R version and package
    versions against r_requirements. Raises SystemExit on mismatch.
    """
    if not r_requirements:
        return

    r_version_minimum: str | None = r_requirements.get("r_version_minimum")
    required_packages: dict[str, str] = r_requirements.get("packages", {})

    # ── Build inline R validation script ──────────────────────────────────────
    pkg_checks = ""
    for pkg_name, required_ver in required_packages.items():
        pkg_checks += textwrap.dedent(f"""\
            pkg_name <- "{pkg_name}"
            required <- "{required_ver}"
            installed <- tryCatch(
              as.character(packageVersion(pkg_name)),
              error = function(e) "NOT_INSTALLED"
            )
            if (installed == "NOT_INSTALLED") {{
              cat("MISSING:", pkg_name, "\\n")
            }} else if (package_version(installed) < package_version(required)) {{
              cat("VERSION_LOW:", pkg_name, installed, "<", required, "\\n")
            }} else {{
              cat("OK:", pkg_name, installed, "\\n")
            }}
        """)

    r_version_check = ""
    if r_version_minimum:
        r_version_check = textwrap.dedent(f"""\
            r_min <- "{r_version_minimum}"
            r_current <- paste0(R.version$major, ".", R.version$minor)
            if (package_version(r_current) < package_version(r_min)) {{
              cat("R_VERSION_LOW:", r_current, "<", r_min, "\\n")
            }} else {{
              cat("R_VERSION_OK:", r_current, "\\n")
            }}
        """)

    script = r_version_check + pkg_checks

    result = subprocess.run(
        ["Rscript", "--vanilla", "-e", script],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"[generate-ground-truth] R validation script failed:\n{result.stderr}"
        )

    output = result.stdout.strip()
    print(f"\n[generate-ground-truth] R environment validation:\n{output}\n")

    errors: list[str] = []
    for line in output.splitlines():
        if line.startswith("R_VERSION_LOW:"):
            errors.append(f"  R version too old: {line.removeprefix('R_VERSION_LOW:').strip()}")
        elif line.startswith("MISSING:"):
            pkg = line.removeprefix("MISSING:").strip()
            errors.append(f"  R package not installed: {pkg}")
        elif line.startswith("VERSION_LOW:"):
            detail = line.removeprefix("VERSION_LOW:").strip()
            errors.append(f"  R package version too low: {detail}")

    if errors:
        raise SystemExit(
            "[generate-ground-truth] R environment does not meet requirements "
            f"defined in {TOX_TOML}:\n" + "\n".join(errors)
        )


def _collect_r_scripts(items: list[pytest.Item]) -> list[Path]:
    """
    Walk collected test items and return unique R script paths (relative
    to TESTS_DIR) extracted from module-level ``r_script`` markers.
    """
    seen: set[Path] = set()
    ordered: list[Path] = []

    for item in items:
        # Module-level pytestmark applies r_script to every item in the module
        marker = item.get_closest_marker("r_script")
        if marker is None:
            continue
        rel_path: str = marker.args[0]
        abs_path = TESTS_DIR / rel_path
        if abs_path not in seen:
            seen.add(abs_path)
            ordered.append(abs_path)
            if not abs_path.exists():
                raise FileNotFoundError(
                    f"[generate-ground-truth] R script declared in test but not found: {abs_path}"
                )

    return ordered


def _run_r_scripts(r_scripts: list[Path]) -> None:
    """
    Generate a temporary master R script that sources every collected R
    script inside an isolated ``local()`` environment, then runs it with
    a single ``Rscript`` invocation (one R session for all scripts).
    """
    source_lines = "\n".join(
        f"  local(source({_r_string(str(p))}, local = TRUE, chdir = FALSE))"
        for p in r_scripts
    )

    master_script = textwrap.dedent(f"""\
        # Auto-generated master R output generation script.
        # Sources every companion R script in a single R session.
        # Working directory is set to the tests/ folder so that
        # relative paths inside R scripts resolve correctly.

        setwd({_r_string(str(TESTS_DIR))})

        run_all <- function() {{
        {source_lines}
        }}

        run_all()
        message("[generate-ground-truth] All R output files generated successfully.")
    """)

    master_path = TESTS_DIR / "_r_master_runner.R"
    try:
        master_path.write_text(master_script, encoding="utf-8")

        print(
            f"[generate-ground-truth] Running {len(r_scripts)} R script(s) "
            "in a single R session..."
        )
        for script in r_scripts:
            print(f"  • {script.relative_to(TESTS_DIR)}")
        print()

        result = subprocess.run(
            ["Rscript", "--vanilla", str(master_path)],
            cwd=str(TESTS_DIR),
            text=True,
        )
        if result.returncode != 0:
            raise SystemExit(
                "[generate-ground-truth] R script execution failed "
                f"(exit code {result.returncode})."
            )
    finally:
        if master_path.exists():
            master_path.unlink()


def _r_string(value: str) -> str:
    """Wrap a Python string as an R string literal (double-quoted, escaped)."""
    escaped = value.replace("\\", "/")  # R prefers forward slashes on all OSes
    return f'"{escaped}"'
