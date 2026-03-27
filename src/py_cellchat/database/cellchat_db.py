from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from typing import Any

import pandas as pd

_VALID_SPECIES = frozenset({"human", "mouse", "zebrafish"})


@dataclass(slots=True)
class CellChatDB:
    interaction_input: pd.DataFrame = field(default_factory=pd.DataFrame)
    complex_input: pd.DataFrame = field(default_factory=pd.DataFrame)
    cofactor_input: pd.DataFrame = field(default_factory=pd.DataFrame)
    gene_info: pd.DataFrame = field(default_factory=pd.DataFrame)
    protein_info: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, species: str) -> CellChatDB:
        """Load a bundled CellChatDB for the given species.

        Parameters
        ----------
        species:
            One of ``"human"``, ``"mouse"``, or ``"zebrafish"``.

        Returns
        -------
        CellChatDB
            A populated CellChatDB instance with interaction, complex,
            cofactor, and gene_info tables loaded from the bundled CSVs.
        """
        species = species.lower()
        if species not in _VALID_SPECIES:
            raise ValueError(
                f"Unknown species '{species}'. "
                f"Valid options are: {sorted(_VALID_SPECIES)}"
            )

        data_pkg = f"py_cellchat.database.data.{species}"

        def _read(filename: str, index_col: str | None) -> pd.DataFrame:
            with resources.files(data_pkg).joinpath(filename).open("r") as fh:
                df = pd.read_csv(fh)
            if index_col is not None and index_col in df.columns:
                df = df.set_index(index_col)
            return df

        interaction = _read("interaction.csv", index_col="interaction_name")
        complex_df = _read("complex.csv", index_col="index")
        cofactor_df = _read("cofactor.csv", index_col="index")
        gene_info = _read("gene_info.csv", index_col=None)

        return cls(
            interaction_input=interaction,
            complex_input=complex_df,
            cofactor_input=cofactor_df,
            gene_info=gene_info,
            metadata={"species": species},
        )
