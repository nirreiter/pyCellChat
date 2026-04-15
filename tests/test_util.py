from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from py_cellchat import CellChat

from typing import Counter


def _approx_equal(a: object, b: object) -> bool:
    """Element-wise equality with 1e-5 absolute tolerance for floats."""
    if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
        return a == pytest.approx(b, abs=1.5e-5, rel=0)
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(_approx_equal(x, y) for x, y in zip(a, b))
    return a == b


def assert_compare(obj1, obj2, is_numeric=False):
	if is_numeric:
		list1 = sorted(obj1)
		list2 = sorted(obj2)
		assert len(list1) == len(list2) 
		equal = [_approx_equal(a, b) for a, b in zip(list1, list2)]
		if not all(equal):
			for i, x in enumerate(equal):
				if not x:
					print("Differs at:", list1[i], list2[i])
			assert False
	else:
		assert Counter(obj1) == Counter(obj2)


# TODO: should be a fixture?
def make_cellchat(adata: ad.AnnData, *, sample_column: str | None = None) -> CellChat:
	return CellChat(
		adata,
		group_by_column="cell_type",
		sample_column=sample_column,
	)


def with_condition_column(adata: ad.AnnData) -> ad.AnnData:
	conditioned = adata.copy()
	groups = conditioned.obs["cell_type"].astype(str)
	condition = pd.Series(index=conditioned.obs_names, dtype="string")

	for group_name in pd.unique(groups):
		group_index = groups.index[groups == group_name]
		labels = np.where(np.arange(len(group_index)) % 2 == 0, "cond_a", "cond_b")
		condition.loc[group_index] = labels

	conditioned.obs["condition"] = pd.Categorical(
		condition,
		categories=["cond_a", "cond_b"],
		ordered=True,
	)
	return conditioned
