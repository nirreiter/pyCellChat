# issues in general logic

Here are the logical inconsistencies and potential issues I found:

---

### 1. Sparse matrix support — critical bug

**R** converts to sparse (`dgCMatrix`) by default and supports sparse input throughout.

**Python** (`__init__`, lines 27–31) raises a `ValueError` for any non-`np.ndarray` type. In practice, `adata.X` is almost always a `scipy.sparse` matrix in AnnData (especially for large datasets), so most real-world inputs will fail with a confusing error.

---

### 2. Matrix orientation mismatch — potential silent bug

**R** stores data as **genes × cells** (`nrow` = genes, `ncol` = cells).

**Python/AnnData** stores data as **cells × genes** (`n_obs` = cells, `n_vars` = genes).

The `__str__` method happens to report the right values because it uses `n_vars`/`n_obs`. However, all downstream computations (mean expression per gene, filtering, etc.) must apply operations on the correct axis. There's no documentation or assertion about this transposition, making it a hidden footgun.

---

# Issues with `identify_ over_expressed_genes`

### 6. Wrong data source — Python uses full data, R uses `data.signaling`

R operates on `object@data.signaling` — a pre-filtered matrix containing **only signaling-relevant genes** (populated by `subsetData()`). Python operates on the full `self.adata`, which includes all genes. This makes the function much slower and changes the semantics: p-value adjustment in R uses `nrow(X)` (total signaling genes) as the Bonferroni denominator, while Python tests all genes.

---

### 7. p-value adjustment method

- **Python**: Benjamini-Hochberg (`method='bh'`)
- **R (slow path)**: Bonferroni, with `n = nrow(X)` — where `n` is the number of **all** signaling genes, not just the tested ones (even stricter than standard Bonferroni)
- **R (fast path / presto)**: BH via `presto`

Python matches the fast-path (presto) behavior, not the slow-path. This discrepancy should be documented.

---

### 8. `no DE` path filters on full `self.adata`, not signaling genes

```python
sc.pp.filter_genes(self.adata, min_cells=min_cells, ...)
```

R's equivalent path filters on `data.use` (which is already `object@data.signaling`). Python skips the feature pre-filtering and runs on all genes, inconsistent with the rest of the function which does subset `adata` to `features`.
