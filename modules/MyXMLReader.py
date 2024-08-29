from pathlib import Path
from typing import Any, Dict, List, Optional
from fsspec import AbstractFileSystem

import pandas as pd
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class MyXMLReader(BaseReader):
    r"""Pandas-based XML parser.

    """

    def __init__(
        self,
        *args: Any,
        concat_rows: bool = True,
        col_joiner: str = ", ",
        row_joiner: str = "\n",
        pandas_config: dict = {},
        **kwargs: Any
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._concat_rows = concat_rows
        self._col_joiner = col_joiner
        self._row_joiner = row_joiner
        self._pandas_config = pandas_config

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
    ) -> List[Document]:
        """Parse file."""
        # open xml file and split it by <sect3> tag
        df = None
        with open(file, "r") as fi:
            data = fi.read().split("<sect3>")
            df = pd.DataFrame(data, columns=["texte"])
        # print the first 5 rows of df
        print(df.head())
        #renama column path_title to document_title
        # df.rename(columns={"path_title": "document_title"}, inplace=True)
        # i put only the text column in the text list
        text_list = df["texte"].tolist()

        #and other columns in the metadata_list
        # metadata_list = df.drop(columns=["texte", "embedding"]).to_dict(orient="records")

        # text_list = df.apply(
        #     lambda row: (self._col_joiner).join(row.astype(str).tolist()), axis=1
        # ).tolist()

        return [
            Document(text=text) for text in text_list
        ]
