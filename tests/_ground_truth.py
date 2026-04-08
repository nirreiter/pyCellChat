"""
This file includes the functionality for parameterizing test functions and generating ground truth cases.

When --generate-ground-truth is passed, it collects whichever tests were selected and determines which r scripts need to be run.
It then produces a single temporary master R script file that includes at least base_ground_truth.R and any other needed R scripts.
It also produces a temporary ground truth manifest file which contains all the test cases, parameterized, so they can be loaded into the R script.
Then the master R script file is run to generate ground truths.
Both temporary files are deleted after generation as long as DELETE_TEMPORARY_FILES is true.
"""

from __future__ import annotations

import json
import subprocess
import textwrap
import tomllib
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pytest


TESTS_DIR = Path(__file__).parent
TOX_TOML = TESTS_DIR.parent / "tox.toml"
GROUND_TRUTH_FILEDIR = TESTS_DIR / "data"
R_GROUND_TRUTH_MANIFEST = TESTS_DIR / "_r_ground_truth_manifest.json"
DELETE_TEMPORARY_FILES = False

MARKER_DOCS = (
    "r_script(path): path (relative to tests/) of the companion R script that generates R output files for this test module.",
    "ground_truth(name): path of the R output file under tests/data/ used by this test.",
    "ground_truth_parameterize(metadata): collection-time metadata for parametrized ground-truth cases. Use via pytest.ground_truth_parameterize(...).",
    "unit: fast unit tests (single-function, no heavy fixtures)",
    "integration: integration tests (end-to-end or multi-component)",
    "synthetic: tests that use small synthetic fixtures (fast)",
    "pbmc3k: tests that use the pbmc3k baseline fixtures (slower)",
)


@dataclass(frozen=True)
class GroundTruthParamSpec:
    r_argument_name: str
    values: list[Any] | None = None
    
    @staticmethod
    def from_tuple(
        arg_name: str,
        raw_config: Any,
    ) -> GroundTruthParamSpec:
        # if isinstance(raw_config, dict):
        #     config = dict(raw_config)
        #     unknown_keys = sorted(config.keys() - {"r_argument_name", "values"})
        #     if unknown_keys:
        #         joined = ", ".join(unknown_keys)
        #         raise pytest.UsageError(
        #             f"pytest.ground_truth_parameterize(...) spec for '{arg_name}' contains "
        #             f"unsupported keys: {joined}."
        #         )
        #     r_argument_name = config.get("r_argument_name", arg_name)
        #     values = config.get("values")
        #     return GroundTruthParamSpec(
        #         r_argument_name=_validated_r_argument_name(arg_name, r_argument_name),
        #         values=_validated_values(arg_name, values),
        #     )

        if isinstance(raw_config, tuple):
            if len(raw_config) == 2:
                r_argument_name, values = raw_config
                
                if not isinstance(r_argument_name, str) or not r_argument_name:
                    raise pytest.UsageError(
                        f"pytest.ground_truth_parameterize(...) r_argument_name for '{arg_name}' "
                        "must be a non-empty string."
                    )
                
                if values is None:
                    value_list = None
                elif isinstance(values, str):
                    raise pytest.UsageError(
                        f"pytest.ground_truth_parameterize(...) values for '{arg_name}' must be a sequence, not a string."
                    )
                else:
                    value_list = list(values)
                    if len(value_list) == 0:
                        raise pytest.UsageError(
                            f"pytest.ground_truth_parameterize(...) values for '{arg_name}' cannot be empty."
                        )
                
                return GroundTruthParamSpec(
                    r_argument_name=r_argument_name,
                    values=value_list,
                )

            # if len(raw_config) == 3:
            #     python_name, r_argument_name, values = raw_config
            #     if python_name != arg_name:
            #         raise pytest.UsageError(
            #             f"pytest.ground_truth_parameterize(...) tuple spec for '{arg_name}' must repeat "
            #             f"the same Python variable name as its first item; got {python_name!r}."
            #         )
            #     return GroundTruthParamSpec(
            #         r_argument_name=_validated_r_argument_name(arg_name, r_argument_name),
            #         values=_validated_values(arg_name, values),
            #     )

            # raise pytest.UsageError(
            #     f"pytest.ground_truth_parameterize(...) tuple spec for '{arg_name}' must have "
            #     "either 2 items: (r_var_name, values_list) or 3 items: "
            #     "(python_var_name, r_var_name, values_list)."
            # )

        raise pytest.UsageError(
            f"pytest.ground_truth_parameterize(...) spec for '{arg_name}' must be "
            "a tuple of the form (r_var_name, values_list)."
        )


def pytest_configure(config: pytest.Config) -> None:
    setattr(pytest, "ground_truth_parameterize", ground_truth_parameterize)
    for marker_doc in MARKER_DOCS:
        config.addinivalue_line("markers", marker_doc)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--generate-ground-truth",
        action="store_true",
        default=False,
        help="Regenerate ground truth files for tests.",
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    marker = metafunc.definition.get_closest_marker("ground_truth_parameterize")
    if marker is None:
        return

    specs = _ground_truth_parameterize_spec(marker)
    argnames = [name for name, spec in specs.items() if spec.values is not None]
    if not argnames:
        return

    overlapping_argnames = sorted(
        set(argnames) & _existing_parametrize_argnames(metafunc.definition)
    )
    if overlapping_argnames:
        joined = ", ".join(overlapping_argnames)
        raise pytest.UsageError(
            "pytest.ground_truth_parameterize(...) cannot also declare values for "
            f"arguments already parametrized with @pytest.mark.parametrize: {joined}."
        )

    value_matrix: list[list[Any]] = []
    for arg_name in argnames:
        values = specs[arg_name].values
        assert values is not None
        value_matrix.append(values)

    cases: list[tuple[object, ...]] = list(product(*value_matrix))
    ids = [_case_id_for_values(tuple(argnames), values) for values in cases]
    metafunc.parametrize(argnames, cases, ids=ids)
    

def _existing_parametrize_argnames(definition: Any) -> set[str]:
    argnames: set[str] = set()
    for marker in definition.iter_markers(name="parametrize"):
        if not marker.args:
            continue
        raw_argnames = marker.args[0]
        if isinstance(raw_argnames, str):
            parts = [name.strip() for name in raw_argnames.split(",")]
        else:
            parts = [str(name).strip() for name in raw_argnames]
        argnames.update(name for name in parts if name)
    return argnames


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if item.get_closest_marker("ground_truth") is not None:
            continue

        marker = item.get_closest_marker("ground_truth_parameterize")
        if marker is None:
            continue

        ground_truth_name = _derived_ground_truth_name(
            item,
            _ground_truth_parameterize_spec(marker),
        )
        item.add_marker(pytest.mark.ground_truth(ground_truth_name))


def pytest_collection_finish(session: pytest.Session) -> None:
    if not session.config.getoption("--generate-ground-truth"):
        return

    r_requirements = _load_r_requirements()
    _validate_r_environment(r_requirements)

    r_scripts = _collect_r_scripts(session.items)
    if not r_scripts:
        print("[generate-ground-truth] No r_script markers found; nothing to generate.")
        return

    GROUND_TRUTH_FILEDIR.mkdir(exist_ok=True)
    _write_r_case_manifest(session.items)

    try:
        _run_r_scripts(r_scripts)
    finally:
        pass
        if DELETE_TEMPORARY_FILES and R_GROUND_TRUTH_MANIFEST.exists():
            R_GROUND_TRUTH_MANIFEST.unlink()


@pytest.fixture
def ground_truth(request: pytest.FixtureRequest) -> Any:
    marker = request.node.get_closest_marker("ground_truth")
    if marker is None:
        pytest.fail(
            "The 'ground_truth' fixture requires @pytest.mark.ground_truth('name') "
            "on the test function."
        )

    names = tuple(str(name) for name in marker.args)
    if len(names) == 1:
        return _load_ground_truth_file(names[0])
    return [_load_ground_truth_file(name) for name in names]


def ground_truth_parameterize(
    metadata: dict[str, Any] | list[tuple[str, str, Any]] | tuple[tuple[str, str, Any], ...] | None = None,
    /,
    **kwargs: Any,
) -> pytest.MarkDecorator:
    if metadata is not None and kwargs:
        raise TypeError(
            "pytest.ground_truth_parameterize accepts either a single metadata mapping "
            "or keyword argument specs, but not both."
        )
    if metadata is None:
        metadata = kwargs
    return pytest.mark.ground_truth_parameterize(metadata)


def _load_ground_truth_file(name: str) -> Any:
    output_path = GROUND_TRUTH_FILEDIR / name
    if not output_path.exists():
        pytest.fail(
            f"Ground truth file not found: {output_path.relative_to(TESTS_DIR.parent)}\n"
            "  Re-run with --generate-ground-truth to create it."
        )

    if output_path.suffix == ".json":
        return json.loads(output_path.read_text(encoding="utf-8"))
    if output_path.suffix == ".h5ad":
        return ad.read_h5ad(output_path)

    pytest.fail(
        f"Invalid extension for ground truth file "
        f"'{output_path.relative_to(TESTS_DIR.parent)}', "
        "only .json and .h5ad files supported!"
    )


def _ground_truth_parameterize_spec(marker: pytest.Mark,) -> dict[str, GroundTruthParamSpec]:
    if not marker.args:
        raise pytest.UsageError(
            "pytest.ground_truth_parameterize(...) requires at least one argument spec."
        )

    raw_spec = marker.args[0]
    # if isinstance(raw_spec, (list, tuple)):
    #     raw_spec = _ground_truth_parameterize_sequence_spec(raw_spec)
    if not isinstance(raw_spec, dict):
        raise pytest.UsageError(
            "pytest.ground_truth_parameterize(...) expects keyword argument specs, a dict, "
            "or a single positional list of tuples."
        )

    specs: dict[str, GroundTruthParamSpec] = {}
    for arg_name, raw_config in raw_spec.items():
        if not isinstance(arg_name, str) or not arg_name:
            raise pytest.UsageError(
                "pytest.ground_truth_parameterize(...) argument names must be non-empty strings."
            )
        specs[arg_name] = GroundTruthParamSpec.from_tuple(arg_name, raw_config)

    return specs


# def _ground_truth_parameterize_sequence_spec(
#     raw_spec: list[tuple[str, str, Any]] | tuple[tuple[str, str, Any], ...],
# ) -> dict[str, tuple[str, Any]]:
#     normalized: dict[str, tuple[str, Any]] = {}
#     for raw_entry in raw_spec:
#         if not isinstance(raw_entry, tuple) or len(raw_entry) != 3:
#             raise pytest.UsageError(
#                 "pytest.ground_truth_parameterize(...) positional sequence entries must be "
#                 "3-item tuples of the form (python_var_name, r_var_name, values_list)."
#             )
#         python_var_name, r_argument_name, values = raw_entry
#         if not isinstance(python_var_name, str) or not python_var_name:
#             raise pytest.UsageError(
#                 "pytest.ground_truth_parameterize(...) positional tuple Python variable names "
#                 "must be non-empty strings."
#             )
#         if python_var_name in normalized:
#             raise pytest.UsageError(
#                 f"pytest.ground_truth_parameterize(...) received duplicate parameter specs for '{python_var_name}'."
#             )
#         normalized[python_var_name] = (r_argument_name, values)
#     return normalized


def _case_id_for_values(argnames: tuple[str, ...], values: tuple[object, ...]) -> str:
    return "-".join(
        f"{arg_name}_{_slugify_case_value(value)}"
        for arg_name, value in zip(argnames, values, strict=True)
    )


def _derived_ground_truth_name(
    item: pytest.Item,
    specs: dict[str, GroundTruthParamSpec],
) -> str:
    if item.get_closest_marker("synthetic") is not None:
        prefix = "synthetic"
    elif item.get_closest_marker("pbmc3k") is not None:
        prefix = "pbmc3k_benchmark"
    else:
        raise pytest.UsageError(
            "pytest.ground_truth_parameterize(...) requires either a synthetic or pbmc3k "
            "marker when no explicit @pytest.mark.ground_truth(...) path is provided."
        )
    
    module_stem = Path(str(item.path)).stem.removeprefix("test_")
    function_folder = module_stem
    for dataset_suffix in ("_synthetic", "_pbmc3k"):
        if module_stem.endswith(dataset_suffix):
            function_folder = module_stem.removesuffix(dataset_suffix)
    
    test_name = getattr(item, "originalname", item.name).split("[", maxsplit=1)[0]
    stem = test_name.removeprefix("test_")

    suffix_parts: list[str] = []
    callspec = getattr(item, "callspec", None)
    if callspec is not None:
        for arg_name, spec in specs.items():
            if spec.values is None or arg_name not in callspec.params:
                continue
            value = callspec.params[arg_name]
            suffix_parts.extend((arg_name, _slugify_case_value(value)))

    suffix = f"_{'_'.join(suffix_parts)}" if suffix_parts else ""
    return f"{prefix}/{function_folder}/{stem}{suffix}.json"


def _slugify_case_value(value: object) -> str:
    normalized = str(value).strip().replace(" ", "_")
    allowed = [char if char.isalnum() or char in {".", "_", "-"} else "_" for char in normalized]
    slug = "".join(allowed).strip("_")
    return slug or "value"


def _load_r_requirements() -> dict[str, Any]:
    if not TOX_TOML.exists():
        print(
            f"\n[generate-ground-truth] WARNING: {TOX_TOML} not found. "
            "R version validation will be skipped."
        )
        return {}

    with TOX_TOML.open("rb") as fh:
        config = tomllib.load(fh)

    reqs = config.get("r_requirements", {})
    if not reqs:
        print(
            "\n[generate-ground-truth] WARNING: No [r_requirements] section in "
            f"{TOX_TOML}. R version validation will be skipped."
        )
    return reqs


def _validate_r_environment(r_requirements: dict[str, Any]) -> None:
    if not r_requirements:
        return

    r_version_minimum = r_requirements.get("r_version_minimum")
    required_packages = r_requirements.get("packages", {})

    pkg_checks = ""
    for pkg_name, required_ver in required_packages.items():
        pkg_checks += textwrap.dedent(f"""\
            pkg_name <- \"{pkg_name}\"
            required <- \"{required_ver}\"
            installed <- tryCatch(
              as.character(packageVersion(pkg_name)),
              error = function(e) \"NOT_INSTALLED\"
            )
            if (installed == \"NOT_INSTALLED\") {{
              cat(\"MISSING:\", pkg_name, \"\\n\")
            }} else if (package_version(installed) < package_version(required)) {{
              cat(\"VERSION_LOW:\", pkg_name, installed, \"<\", required, \"\\n\")
            }} else {{
              cat(\"OK:\", pkg_name, installed, \"\\n\")
            }}
        """)

    r_version_check = ""
    if r_version_minimum:
        r_version_check = textwrap.dedent(f"""\
            r_min <- \"{r_version_minimum}\"
            r_current <- paste0(R.version$major, \".\", R.version$minor)
            if (package_version(r_current) < package_version(r_min)) {{
              cat(\"R_VERSION_LOW:\", r_current, \"<\", r_min, \"\\n\")
            }} else {{
              cat(\"R_VERSION_OK:\", r_current, \"\\n\")
            }}
        """)

    result = subprocess.run(
        ["Rscript", "--vanilla", "-e", r_version_check + pkg_checks],
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
            errors.append(f"  R package not installed: {line.removeprefix('MISSING:').strip()}")
        elif line.startswith("VERSION_LOW:"):
            errors.append(f"  R package version too low: {line.removeprefix('VERSION_LOW:').strip()}")

    if errors:
        raise SystemExit(
            "[generate-ground-truth] R environment does not meet requirements "
            f"defined in {TOX_TOML}:\n" + "\n".join(errors)
        )


def _collect_r_scripts(items: list[pytest.Item]) -> list[Path]:
    seen: set[Path] = set()
    ordered: list[Path] = []

    for item in items:
        marker = item.get_closest_marker("r_script")
        if marker is None:
            continue
        abs_path = TESTS_DIR / str(marker.args[0])
        if abs_path in seen:
            continue
        if not abs_path.exists():
            raise FileNotFoundError(
                f"[generate-ground-truth] R script declared in test but not found: {abs_path}"
            )
        seen.add(abs_path)
        ordered.append(abs_path)

    return ordered


def _write_r_case_manifest(items: list[pytest.Item]) -> None:
    cases: list[dict[str, Any]] = []

    for item in items:
        r_script_marker = item.get_closest_marker("r_script")
        ground_truth_marker = item.get_closest_marker("ground_truth")
        if r_script_marker is None or ground_truth_marker is None:
            continue

        parameterize_marker = item.get_closest_marker("ground_truth_parameterize")
        specs = (
            _ground_truth_parameterize_spec(parameterize_marker)
            if parameterize_marker is not None
            else {}
        )
        params = _case_params(item, specs)
        case_id = getattr(getattr(item, "callspec", None), "id", None) or item.name
        cases.append(
            {
                "nodeid": item.nodeid,
                "case_id": case_id,
                "r_script": str(r_script_marker.args[0]),
                "ground_truth": [str(name) for name in ground_truth_marker.args],
                "params": params,
            }
        )

    R_GROUND_TRUTH_MANIFEST.write_text(
        json.dumps({"cases": cases}, indent=2),
        encoding="utf-8",
    )


def _case_params(
    item: pytest.Item,
    specs: dict[str, GroundTruthParamSpec],
) -> dict[str, Any]:
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return {}

    params: dict[str, Any] = {}
    for arg_name, spec in specs.items():
        if arg_name not in callspec.params:
            continue
        params[spec.r_argument_name] = _json_compatible(callspec.params[arg_name])
    return params


def _json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)

    try:
        json.dumps(value)
    except TypeError as exc:
        raise TypeError(
            f"Ground-truth parameter value {value!r} is not JSON-serializable."
        ) from exc
    return value


def _run_r_scripts(r_scripts: list[Path]) -> None:
    source_lines = "\n".join(
        f"  local(source({_r_string(str(path))}, local = TRUE, chdir = FALSE))"
        for path in r_scripts
    )

    master_script = textwrap.dedent(f"""\
        # Auto-generated master R output generation script.
        # Sources every companion R script in a single R session.

        setwd({_r_string(str(TESTS_DIR))})
        options(show.error.locations = TRUE)
        options(error = function() {{
            cat("[generate-ground-truth] R error occurred. Traceback:\n")
            traceback()
            quit(status = 1)
        }})

        run_all <- function() {{
        {source_lines}
        }}

        run_all()
        warnings()
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
        if DELETE_TEMPORARY_FILES and master_path.exists():
            master_path.unlink()


def _r_string(value: str) -> str:
    escaped = value.replace("\\", "/").replace('"', '\\"')
    return f'"{escaped}"'
