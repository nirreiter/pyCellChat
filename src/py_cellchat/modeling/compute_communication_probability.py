from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import sparse

from ..core.matrix import get_adata_matrix_checked
from .expression import (
    compute_expr_agonist,
    compute_expr_antagonist,
    compute_expr_coreceptor,
    compute_expr_lr,
)
from .statistics import build_group_average

if TYPE_CHECKING:
    from ..core.cellchat import CellChat


_ANNOTATION_ORDER = [
    "Secreted Signaling",
    "ECM-Receptor",
    "Non-protein Signaling",
    "Cell-Cell Contact",
]

_R_UINT32_MASK = 0xFFFFFFFF
_R_MT_STATE_SIZE = 624
_R_MT_PERIOD = 397
_R_MT_MATRIX_A = 0x9908B0DF
_R_MT_UPPER_MASK = 0x80000000
_R_MT_LOWER_MASK = 0x7FFFFFFF
_R_MT_TEMPERING_MASK_B = 0x9D2C5680
_R_MT_TEMPERING_MASK_C = 0xEFC60000
_R_I2_32M1 = 2.328306437080797e-10


def compute_communication_probability(
    cellchat: CellChat,
    type: str = "triMean",
    trim: float = 0.1,
    lr_use: pd.DataFrame | None = None,
    raw_use: bool = True,
    population_size: bool = False,
    nboot: int = 100,
    seed_use: int = 1,
    Kh: float = 0.5,
    n: float = 1,
    # parameters for spatial datasets:
    distance_use: bool = True,
    interaction_range: float = 250,
    scale_distance: float = 0.01,
    k_min: int = 10,
    contact_dependent: bool = True,
    contact_range: float | None = None,
    contact_knn_k: int | None = None,
    contact_dependent_forced: bool = False,
    do_symmetric: bool = True,
) -> CellChat:
    if cellchat.db is None:
        raise ValueError("Must load a CellChatDB before running compute_communication_probability")
    if cellchat.adata_signaling is None:
        raise ValueError("Must run subset_data on the CellChat object first")
    if cellchat.options.get("datatype") != "RNA":
        raise NotImplementedError(
            "Only RNA Datsets currently supported. Spatial compute_communication_probability is not implemented yet"
        )
    if not raw_use:
        raise NotImplementedError(
            "raw_use=False requires projected or smoothed data support, which is not implemented yet"
        )
    if nboot <= 0:
        raise ValueError("nboot must be a positive integer")

    group = cellchat.idents
    group_levels = _group_levels(group)
    pair_lrsig = _resolve_pair_lr_use(cellchat, lr_use)
    n_lr = len(pair_lrsig)
    n_groups = len(group_levels)

    if n_lr == 0:
        cellchat.net = {
            "prob": np.zeros((n_groups, n_groups, 0), dtype=float),
            "pval": np.zeros((n_groups, n_groups, 0), dtype=float),
        }
        return cellchat

    matrix = get_adata_matrix_checked(cellchat.adata_signaling, False, None)
    data = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    data = np.asarray(data, dtype=float)
    data_max = float(np.max(data)) if data.size else 0.0
    data_use = data if data_max == 0 else data / data_max

    gene_names = cellchat.adata_signaling.var_names.astype(str).tolist()
    mean_function = build_group_average(type, trim=trim)
    group_codes = _group_codes(group, group_levels)
    started = perf_counter()
    data_use_avg = _aggregate_expression_by_group(
        data_use,
        gene_names,
        group_codes,
        group_levels,
        mean_function,
    )

    gene_l = pair_lrsig["ligand"].astype(str).tolist()
    gene_r = pair_lrsig["receptor"].astype(str).tolist()
    data_lavg = compute_expr_lr(gene_l, data_use_avg, cellchat.db.complex)
    data_ravg = compute_expr_lr(gene_r, data_use_avg, cellchat.db.complex)
    data_ravg_co_a = compute_expr_coreceptor(cellchat.db.cofactor, data_use_avg, pair_lrsig, receptor_type="A")
    data_ravg_co_i = compute_expr_coreceptor(cellchat.db.cofactor, data_use_avg, pair_lrsig, receptor_type="I")
    data_ravg = data_ravg * data_ravg_co_a / data_ravg_co_i

    group_sizes = group.value_counts(sort=False).reindex(group_levels).to_numpy(dtype=float)
    group_props = group_sizes / len(group)
    population_outer = np.outer(group_props, group_props)
    ones = np.ones((n_groups, n_groups), dtype=float)

    index_agonist = set(_nonempty_row_indices(pair_lrsig, "agonist"))
    index_antagonist = set(_nonempty_row_indices(pair_lrsig, "antagonist"))
    prob = np.zeros((n_groups, n_groups, n_lr), dtype=float)
    pval = np.zeros((n_groups, n_groups, n_lr), dtype=float)
    
    bootstrap_rng = _RBootstrapSampler(seed_use)
    permutations = [bootstrap_rng.permutation(len(group)) for _ in range(nboot)]
    boot_averages = [
        _aggregate_expression_by_group(
            data_use,
            gene_names,
            group_codes[permutation],
            group_levels,
            mean_function,
        )
        for permutation in permutations
    ]

    for idx in range(n_lr):
        p1 = _hill_outer(data_lavg[idx, :], data_ravg[idx, :], Kh=Kh, hill_n=n)
        if float(np.sum(p1)) == 0.0:
            pval[:, :, idx] = 1.0
            continue

        p2 = ones
        if idx in index_agonist:
            agonist = compute_expr_agonist(data_use_avg, pair_lrsig, cellchat.db.cofactor, idx, kh=Kh, hill_n=n)
            p2 = np.outer(agonist, agonist)

        p3 = ones
        if idx in index_antagonist:
            antagonist = compute_expr_antagonist(data_use_avg, pair_lrsig, cellchat.db.cofactor, idx, kh=Kh, hill_n=n)
            p3 = np.outer(antagonist, antagonist)

        p4 = population_outer if population_size else ones
        observed = p1 * p2 * p3 * p4
        prob[:, :, idx] = observed
        observed_flat = observed.reshape(-1)
        
        boot_values = np.empty((observed_flat.size, nboot), dtype=float)
        for boot_index, boot_avg in enumerate(boot_averages):
            boot_lavg = compute_expr_lr([gene_l[idx]], boot_avg, cellchat.db.complex)
            boot_ravg = compute_expr_lr([gene_r[idx]], boot_avg, cellchat.db.complex)
            boot_ravg = boot_ravg * compute_expr_coreceptor(
                cellchat.db.cofactor,
                boot_avg,
                pair_lrsig.iloc[[idx]],
                receptor_type="A",
            ) / compute_expr_coreceptor(
                cellchat.db.cofactor,
                boot_avg,
                pair_lrsig.iloc[[idx]],
                receptor_type="I",
            )
            boot_p1 = _hill_outer(boot_lavg[0, :], boot_ravg[0, :], Kh=Kh, hill_n=n)

            boot_p2 = ones
            if idx in index_agonist:
                agonist = compute_expr_agonist(boot_avg, pair_lrsig, cellchat.db.cofactor, idx, kh=Kh, hill_n=n)
                boot_p2 = np.outer(agonist, agonist)

            boot_p3 = ones
            if idx in index_antagonist:
                antagonist = compute_expr_antagonist(boot_avg, pair_lrsig, cellchat.db.cofactor, idx, kh=Kh, hill_n=n)
                boot_p3 = np.outer(antagonist, antagonist)

            boot_p4 = population_outer if population_size else ones
            boot_values[:, boot_index] = (boot_p1 * boot_p2 * boot_p3 * boot_p4).reshape(-1)

        rejected = np.sum(boot_values > observed_flat[:, None], axis=1)
        pval[:, :, idx] = rejected.reshape(n_groups, n_groups) / nboot

    pval[prob == 0] = 1.0
    cellchat.net = {
        "prob": prob,
        "pval": pval,
        "pair_lr_use": pair_lrsig.copy(),
    }
    cellchat.options["parameter"] = {
        "type_mean": type,
        "trim": trim,
        "raw_use": raw_use,
    }
    cellchat.options["run.time"] = float(perf_counter() - started)
    return cellchat


def _group_levels(group: pd.Series) -> list[str]:
    if isinstance(group.dtype, pd.CategoricalDtype):
        categories = group.cat.categories.astype(str).tolist()
        present = pd.unique(group.astype(str)).tolist()
        if len(categories) != len(present):
            raise ValueError(
                "Please check unique(cellchat.idents) and ensure that the factor levels are correct. "
                "You may need to drop unused levels before running compute_communication_probability."
            )
        return categories
    return pd.unique(group.astype(str)).tolist()


def _resolve_pair_lr_use(cellchat: CellChat, lr_use: pd.DataFrame | None) -> pd.DataFrame:
    if lr_use is None:
        pair_lr_use = cellchat.lr if cellchat.lr is not None else cellchat.db.interaction
    else:
        pair_lr_use = lr_use

    pair_lr_use = pair_lr_use.copy()
    if "annotation" in pair_lr_use.columns and pair_lr_use["annotation"].nunique() > 1:
        dtype = pd.CategoricalDtype(categories=_ANNOTATION_ORDER, ordered=True)
        pair_lr_use = pair_lr_use.assign(
            annotation=pair_lr_use["annotation"].astype(dtype)
        ).sort_values("annotation", kind="stable")
        pair_lr_use["annotation"] = pair_lr_use["annotation"].astype(str)
    return pair_lr_use


def _group_codes(group: pd.Series, group_levels: Sequence[str]) -> np.ndarray:
    dtype = pd.CategoricalDtype(categories=list(group_levels), ordered=True)
    codes = group.astype(dtype).cat.codes.to_numpy(dtype=int, copy=False)
    if np.any(codes < 0):
        raise ValueError("group contains values outside the declared group levels")
    return codes


def _aggregate_expression_by_group(
    data_use: np.ndarray,
    gene_names: Sequence[str],
    group_codes: np.ndarray,
    group_levels: Sequence[str],
    mean_function,
) -> pd.DataFrame:
    aggregated = np.zeros((len(gene_names), len(group_levels)), dtype=float)
    for idx, _level in enumerate(group_levels):
        mask = group_codes == idx
        group_matrix = data_use[mask, :]
        if group_matrix.shape[0] == 0:
            continue
        aggregated[:, idx] = np.asarray(mean_function(group_matrix), dtype=float)
    return pd.DataFrame(aggregated, index=list(gene_names), columns=list(group_levels))


def _hill_outer(ligand: np.ndarray, receptor: np.ndarray, Kh: float, hill_n: float) -> np.ndarray:
    interaction = np.outer(np.asarray(ligand, dtype=float), np.asarray(receptor, dtype=float))
    i_hn = np.power(interaction, hill_n)
    return i_hn / (np.power(Kh, hill_n) + i_hn)


def _nonempty_row_indices(frame: pd.DataFrame, column_name: str) -> list[int]:
    if column_name not in frame.columns:
        return []
    values = frame[column_name]
    return [
        idx
        for idx, value in enumerate(values.tolist())
        if isinstance(value, str) and value != ""
    ]

# This class exists to mimic R's RNG behavior for bootstrap sampling.
# The goal is reproducibility against upstream CellChat, not performance.
class _RBootstrapSampler:
    """R-compatible RNG for bootstrap permutations used by computeCommunProb."""

    def __init__(self, seed: int) -> None:
        # Start from the user-provided seed and force it into 32-bit unsigned space,
        # because R's MT implementation operates on 32-bit integers.
        scrambled = int(seed) & _R_UINT32_MASK

        # Apply the same linear scrambling repeatedly to diffuse the seed
        # before filling the MT state array.
        for _ in range(50):
            scrambled = (69069 * scrambled + 1) & _R_UINT32_MASK

        # Allocate MT state.
        # Convention here:
        # - self._state[0] stores the current index into the MT array
        # - self._state[1:] stores the 624 MT state words
        self._state = [0] * (_R_MT_STATE_SIZE + 1)

        # Fill the MT state using the same recurrence.
        for index in range(_R_MT_STATE_SIZE + 1):
            scrambled = (69069 * scrambled + 1) & _R_UINT32_MASK
            self._state[index] = scrambled

        # Initialize the "current index" to 624 so the first draw triggers a twist step.
        self._state[0] = _R_MT_STATE_SIZE

    def permutation(self, size: int) -> np.ndarray:
        # Build a random permutation of [0, 1, ..., size-1].
        # This is effectively sampling without replacement.
        remaining = list(range(size))
        out = np.empty(size, dtype=int)
        n_remaining = size

        for index in range(size):
            # Draw one unbiased random index in [0, n_remaining).
            selected = self._unif_index(n_remaining)
            out[index] = remaining[selected]

            # Remove the selected value by swapping the tail element into its place.
            n_remaining -= 1
            remaining[selected] = remaining[n_remaining]

        return out

    def _unif_rand(self) -> float:
        # Produce an R-style uniform random number in the open interval (0, 1),
        # avoiding exact 0 and exact 1.
        return _fixup_r_uniform(self._mt_genrand())

    def _mt_genrand(self) -> float:
        # Pull the current MT state and index.
        mt = self._state[1:]
        index = self._state[0]

        # If we've exhausted the current state, generate the next 624 values.
        # This is the "twist" step of the Mersenne Twister algorithm.
        if index >= _R_MT_STATE_SIZE:
            for kk in range(_R_MT_STATE_SIZE - _R_MT_PERIOD):
                value = (mt[kk] & _R_MT_UPPER_MASK) | (mt[kk + 1] & _R_MT_LOWER_MASK)
                mt[kk] = (
                    mt[kk + _R_MT_PERIOD]
                    ^ (value >> 1)
                    ^ (_R_MT_MATRIX_A if value & 1 else 0)
                ) & _R_UINT32_MASK

            for kk in range(_R_MT_STATE_SIZE - _R_MT_PERIOD, _R_MT_STATE_SIZE - 1):
                value = (mt[kk] & _R_MT_UPPER_MASK) | (mt[kk + 1] & _R_MT_LOWER_MASK)
                mt[kk] = (
                    mt[kk + (_R_MT_PERIOD - _R_MT_STATE_SIZE)]
                    ^ (value >> 1)
                    ^ (_R_MT_MATRIX_A if value & 1 else 0)
                ) & _R_UINT32_MASK

            # Handle the wraparound case for the last MT entry.
            value = (mt[_R_MT_STATE_SIZE - 1] & _R_MT_UPPER_MASK) | (mt[0] & _R_MT_LOWER_MASK)
            mt[_R_MT_STATE_SIZE - 1] = (
                mt[_R_MT_PERIOD - 1]
                ^ (value >> 1)
                ^ (_R_MT_MATRIX_A if value & 1 else 0)
            ) & _R_UINT32_MASK

            index = 0

        # Extract the next raw MT value.
        value = mt[index]
        index += 1

        # Tempering step:
        # this is the standard MT output transformation that improves bit quality.
        value ^= value >> 11
        value ^= (value << 7) & _R_MT_TEMPERING_MASK_B
        value ^= (value << 15) & _R_MT_TEMPERING_MASK_C
        value ^= value >> 18

        # Write back updated state.
        self._state[0] = index
        self._state[1:] = mt

        # Convert the 32-bit integer into a float in roughly [0, 1).
        return (value & _R_UINT32_MASK) * 2.3283064365386963e-10

    def _unif_index(self, size: int) -> int:
        # Defensive fallback; not expected in normal permutation use.
        if size <= 0:
            return 0

        # Compute the number of bits needed to represent integers < size.
        bits = int(np.ceil(np.log2(size)))

        # Rejection sampling:
        # draw a candidate in [0, 2^bits), and reject it if it's outside [0, size).
        # This avoids modulo bias.
        while True:
            value = self._random_bits(bits)
            if value < size:
                return value

    def _random_bits(self, bits: int) -> int:
        # Build up at least `bits` random bits using 16-bit chunks derived from
        # uniform floats, mirroring R's style of generating indices.
        value = 0
        current = 0
        while current <= bits:
            value = (65536 * value) + int(np.floor(self._unif_rand() * 65536.0))
            current += 16

        # Keep only the requested number of low bits.
        return value & ((1 << bits) - 1)


def _fixup_r_uniform(value: float) -> float:
    if value <= 0.0:
        return 0.5 * _R_I2_32M1
    if (1.0 - value) <= 0.0:
        return 1.0 - (0.5 * _R_I2_32M1)
    return value
