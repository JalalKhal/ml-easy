import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    IO,
    Callable,
    Generic,
    Self,
    TypeVar, Iterable
)

import pandas as pd
import polars as pl


_logger = logging.getLogger(__name__)

V = TypeVar('V', bound = "Iterable")


class Dataset(ABC, Generic[V]):

    def __init__(self, service: V):
        self.service = service

    def __iter__(self) -> Iterable:
        return self.collect().service.__iter__()

    def collect(self) -> Self:
        return self

    @abstractmethod
    def to_pandas(self) -> pd.DataFrame:
        pass

    @classmethod
    @abstractmethod
    def read_csv(
            cls,
            source: str | Path | IO[str] | IO[bytes] | bytes,
            separator: str,
            encoding: str = 'utf8',
            transform_columns: Callable[[str], str] = lambda x: x,
    ) -> Self:
        pass

    @abstractmethod
    def write_csv(
            self,
            file: str,
            separator: str = ',',
    ) -> None:
        pass


class PolarsDataset(Dataset[pl.DataFrame | pl.LazyFrame]):
    def __init__(
            self,
            service: pl.DataFrame | pl.LazyFrame
    ):
        super().__init__(service)
    def to_pandas(self):
        return self.get_dataframe().to_pandas()

    def get_dataframe(self) -> pl.DataFrame:
        df = self.service.collect() if isinstance(self.service, pl.LazyFrame) else self.service
        self.service = df.lazy()
        return df

    @classmethod
    def read_csv(
            cls,
            source: str | Path | IO[str] | IO[bytes] | bytes,
            separator: str,
            encoding: str = 'utf8',
            transform_columns: Callable[[str], str] = lambda x: x,
    ) -> Self:
        ds = (
            pl.read_csv(
                source,
                separator=separator,
                encoding=encoding,
            )
            .rename(transform_columns)
            .lazy()
        )
        return cls(service=ds)

    def write_csv(
            self,
            file: str,
            separator: str = ',',
    ) -> None:
        self.service.collect().write_csv(file, separator = separator)

    def collect(self) -> Self:
        return PolarsDataset(self.service.collect())


