from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .statistics import geometric_mean


def compute_expr_complex(
    complex_input: pd.DataFrame,
    data_use: pd.DataFrame,
    complex_names: Sequence[str],
) -> np.ndarray:
    result = np.zeros((len(complex_names), data_use.shape[1]), dtype=float)
    if complex_input.empty:
        return result

    subunit_cols = [column for column in complex_input.columns if column.startswith("subunit")]
    for row_index, complex_name in enumerate(complex_names):
        if complex_name not in complex_input.index:
            continue
        subunits = [
            value
            for value in complex_input.loc[complex_name, subunit_cols].tolist()
            if isinstance(value, str) and value != "" and value in data_use.index
        ]
        if not subunits:
            continue
        result[row_index, :] = np.asarray(
            geometric_mean(data_use.loc[subunits].to_numpy(dtype=float)),
            dtype=float,
        )
    return result


def compute_expr_lr(
    gene_lr: Sequence[str],
    data_use: pd.DataFrame,
    complex_input: pd.DataFrame,
) -> np.ndarray:
    n_lr = len(gene_lr)
    result = np.zeros((n_lr, data_use.shape[1]), dtype=float)

    complex_names: list[str] = []
    complex_indices: list[int] = []
    for idx, gene_name in enumerate(gene_lr):
        if gene_name in data_use.index:
            result[idx, :] = data_use.loc[gene_name].to_numpy(dtype=float)
            continue
        complex_names.append(gene_name)
        complex_indices.append(idx)

    if complex_names:
        complex_values = compute_expr_complex(complex_input, data_use, complex_names)
        for idx, values in zip(complex_indices, complex_values, strict=False):
            result[idx, :] = values

    return result


def compute_expr_coreceptor(
    cofactor_input: pd.DataFrame,
    data_use: pd.DataFrame,
    pair_lrsig: pd.DataFrame,
    receptor_type: str = "A",
) -> np.ndarray:
    n_pairs = len(pair_lrsig)
    result = np.ones((n_pairs, data_use.shape[1]), dtype=float)
    column_name = "co_A_receptor" if receptor_type == "A" else "co_I_receptor"
    if column_name not in pair_lrsig.columns:
        return result

    for row_index, cofactor_name in enumerate(pair_lrsig[column_name].tolist()):
        genes = _cofactor_genes(cofactor_input, data_use, cofactor_name)
        if len(genes) == 1:
            result[row_index, :] = 1.0 + data_use.loc[genes[0]].to_numpy(dtype=float)
        elif len(genes) > 1:
            result[row_index, :] = np.prod(1.0 + data_use.loc[genes].to_numpy(dtype=float), axis=0)
    return result


def compute_expr_agonist(
    data_use: pd.DataFrame,
    pair_lrsig: pd.DataFrame,
    cofactor_input: pd.DataFrame,
    index_agonist: int,
    kh: float,
    hill_n: float,
) -> np.ndarray:
    if "agonist" not in pair_lrsig.columns:
        return np.ones(data_use.shape[1], dtype=float)

    genes = _cofactor_genes(cofactor_input, data_use, pair_lrsig.iloc[index_agonist]["agonist"])
    if not genes:
        return np.ones(data_use.shape[1], dtype=float)

    values = data_use.loc[genes].to_numpy(dtype=float)
    activated = 1.0 + np.power(values, hill_n) / (np.power(kh, hill_n) + np.power(values, hill_n))
    if len(genes) == 1:
        return activated[0]
    return np.prod(activated, axis=0)


def compute_expr_antagonist(
    data_use: pd.DataFrame,
    pair_lrsig: pd.DataFrame,
    cofactor_input: pd.DataFrame,
    index_antagonist: int,
    kh: float,
    hill_n: float,
) -> np.ndarray:
    if "antagonist" not in pair_lrsig.columns:
        return np.ones(data_use.shape[1], dtype=float)

    genes = _cofactor_genes(cofactor_input, data_use, pair_lrsig.iloc[index_antagonist]["antagonist"])
    if not genes:
        return np.ones(data_use.shape[1], dtype=float)

    values = data_use.loc[genes].to_numpy(dtype=float)
    suppressed = np.power(kh, hill_n) / (np.power(kh, hill_n) + np.power(values, hill_n))
    if len(genes) == 1:
        return suppressed[0]
    return np.prod(suppressed, axis=0)


def _cofactor_genes(
    cofactor_input: pd.DataFrame,
    data_use: pd.DataFrame,
    cofactor_name: object,
) -> list[str]:
    if not isinstance(cofactor_name, str) or cofactor_name == "":
        return []
    if cofactor_input.empty or cofactor_name not in cofactor_input.index:
        return []

    cofactor_cols = [column for column in cofactor_input.columns if column.startswith("cofactor")]
    if not cofactor_cols:
        return []

    row = cofactor_input.loc[cofactor_name, cofactor_cols]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return [
        value
        for value in row.tolist()
        if isinstance(value, str) and value != "" and value in data_use.index
    ]
