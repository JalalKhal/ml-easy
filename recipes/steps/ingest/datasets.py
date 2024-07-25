import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    IO,
    Callable,
    Generic,
    Self,
    TypeVar, Iterable, List, Tuple, Any
)

import numpy as np
import pandas as pd
import polars as pl

_logger = logging.getLogger(__name__)

V = TypeVar('V')


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

    @abstractmethod
    def split(self, train_prop: float, val_prop: float) -> Tuple[Self, Self, Self]:
        pass

    @abstractmethod
    def to_numpy(self):
        pass

    @classmethod
    @abstractmethod
    def from_numpy(
            self,
            data: np.ndarray[Any, Any],
    ) -> Self:
        pass

    @abstractmethod
    def select(self, cols: List[str]) -> Self:
        pass

    @abstractmethod
    def columns(self) -> List[str]:
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
        self.service.collect().write_csv(file, separator=separator)

    def collect(self) -> Self:
        return PolarsDataset(self.service.collect())

    def split(self, train_prop: float, val_prop: float) -> Tuple[Self, Self, Self]:
        total_samples = self.service.select(pl.len()).collect().item()
        train_size = int(train_prop * total_samples)
        val_size = int(val_prop * total_samples)
        test_size = total_samples - train_size - val_size

        # Split the dataframe
        test_df = self.__class__(self.service.head(test_size))
        val_df = self.__class__(self.service.slice(test_size, val_size))
        train_df = self.__class__(self.service.tail(train_size))
        return train_df, val_df, test_df

    def to_numpy(self) -> np.ndarray[Any, Any]:
        return self.get_dataframe().to_numpy()

    def from_numpy(
            self,
            data: np.ndarray[Any, Any],
    ) -> Self:
        return self.__class__(pl.from_numpy(data))

    def select(self, cols: List[str]) -> Self:
        return self.__class__(service=self.get_dataframe().select(cols))

    def columns(self) -> List[str]:
        return self.service.columns

