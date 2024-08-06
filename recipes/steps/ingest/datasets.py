import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    IO,
    Any,
    Callable,
    Collection,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Self,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd
import polars as pl
from polars._typing import ColumnNameOrSelector, ConcatMethod

from recipes.enum import MLFlowErrorCode
from recipes.exceptions import MlflowException
from recipes.steps.transform.filters import EqualFilter, InFilter

_logger = logging.getLogger(__name__)

V = TypeVar('V')


class Dataset(ABC, Generic[V]):

    def __init__(self, service: V):
        self.service = service

    @abstractmethod
    def __iter__(self) -> Iterable:
        pass

    @abstractmethod
    def to_pandas(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def split(self, train_prop: float, val_prop: float) -> Tuple[Self, Self, Self]:
        pass

    @abstractmethod
    def to_numpy(self) -> np.ndarray[Any, Any]:
        pass

    @classmethod
    @abstractmethod
    def from_numpy(
        cls,
        data: np.ndarray[Any, Any],
    ) -> Self:
        pass

    @abstractmethod
    def select(self, cols: List[str]) -> Self:
        pass

    @abstractmethod
    def columns(self) -> List[str]:
        pass

    @abstractmethod
    def collect(self) -> Self:
        pass

    @abstractmethod
    def filter(self, filters: Dict[str, Optional[Union[EqualFilter[str], InFilter[str]]]]) -> Self:
        pass

    @abstractmethod
    def drop_nulls(
        self,
        subset: ColumnNameOrSelector | Collection[ColumnNameOrSelector] | None = None,
    ) -> Self:
        pass

    @classmethod
    @abstractmethod
    def concat(
        cls, items: Iterable[Self], *, how: ConcatMethod = 'vertical', rechunk: bool = False, parallel: bool = True
    ) -> Self:
        pass

    @abstractmethod
    def concatenate(
        self, items: Iterable[Self], *, how: ConcatMethod = 'vertical', rechunk: bool = False, parallel: bool = True
    ) -> Self:
        pass

    @abstractmethod
    def write_csv(
        self,
        file: str | Path | IO[str] | IO[bytes] | None = None,
        *,
        include_bom: bool = False,
        include_header: bool = True,
        separator: str = ',',
        line_terminator: str = '\n',
        quote_char: str = '"',
        batch_size: int = 1024,
        datetime_format: str | None = None,
        date_format: str | None = None,
        time_format: str | None = None,
        float_precision: int | None = None,
        null_value: str | None = None,
    ) -> str | None:
        pass

    @abstractmethod
    def slice(self, offset: int, length: int | None = None) -> Self:
        pass


class PolarsDataset(Dataset[pl.DataFrame | pl.LazyFrame]):
    def __init__(self, service: pl.DataFrame | pl.LazyFrame):
        super().__init__(service)

    def to_pandas(self):
        return self.get_dataframe().to_pandas()

    def get_dataframe(self) -> pl.DataFrame:
        df = self.service.collect() if isinstance(self.service, pl.LazyFrame) else self.service
        return df

    def __iter__(self) -> Iterable:
        ds: pl.DataFrame = self.get_dataframe()
        return ds.__iter__()

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
        file: str | Path | IO[str] | IO[bytes] | None = None,
        *,
        include_bom: bool = False,
        include_header: bool = True,
        separator: str = ',',
        line_terminator: str = '\n',
        quote_char: str = '"',
        batch_size: int = 1024,
        datetime_format: str | None = None,
        date_format: str | None = None,
        time_format: str | None = None,
        float_precision: int | None = None,
        null_value: str | None = None,
    ) -> str | None:
        return self.get_dataframe().write_csv(
            file,
            include_bom=include_bom,
            include_header=include_header,
            separator=separator,
            line_terminator=line_terminator,
            quote_char=quote_char,
            batch_size=batch_size,
            date_format=date_format,
            datetime_format=datetime_format,
            time_format=time_format,
            float_precision=float_precision,
            null_value=null_value,
        )

    def split(self, train_prop: float, val_prop: float) -> Tuple[Self, Self, Self]:
        total_samples = self.get_dataframe().select(pl.len()).item()
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

    @classmethod
    def from_numpy(
        cls,
        data: np.ndarray[Any, Any],
    ) -> Self:
        return cls(pl.from_numpy(data))

    @classmethod
    def concat(
        cls, items: Iterable[Self], *, how: ConcatMethod = 'vertical', rechunk: bool = False, parallel: bool = True
    ) -> Self:
        return cls(pl.concat([it.service for it in items], how=how, rechunk=rechunk, parallel=parallel))  # type: ignore

    def concatenate(
        self, items: Iterable[Self], *, how: ConcatMethod = 'vertical', rechunk: bool = False, parallel: bool = True
    ) -> Self:
        return self.__class__.concat([self] + items, how='horizontal', rechunk=rechunk, parallel=parallel)

    def select(self, cols: List[str]) -> Self:
        return self.__class__(service=self.get_dataframe().select(cols))

    def columns(self) -> List[str]:
        return self.get_dataframe().columns

    def collect(self) -> Self:
        return self.__class__(self.get_dataframe())

    def drop_nulls(
        self,
        subset: ColumnNameOrSelector | Collection[ColumnNameOrSelector] | None = None,
    ) -> Self:
        return self.__class__(self.service.drop_nulls(subset))

    def filter(self, filters: Dict[str, Optional[List[Union[EqualFilter[str], InFilter[str]]]]]) -> Self:
        from recipes.utils import is_instance_for_generic

        def _get_expr_filter(col_filter: Union[EqualFilter[str], InFilter[str]]):
            if is_instance_for_generic(col_filter, EqualFilter[str]):
                return pl.col(col) == col_filter.value  # type:ignore
            elif is_instance_for_generic(col_filter, InFilter[str]):
                return pl.col(col).is_in(col_filter.values)  # type:ignore
            else:
                raise MlflowException(
                    message=f'Unsupported filter type {col_filter.__class__.__name__}',
                    error_code=MLFlowErrorCode.INVALID_PARAMETER_VALUE,
                )

        predicates_expr = []
        for col in filters:
            col_filters = filters[col]
            for filter in col_filters:
                predicates_expr.append(_get_expr_filter(filter))

        return self.__class__(self.service.filter(predicates_expr))

    def slice(self, offset: int, length: int | None = None) -> Self:
        return self.__class__(self.get_dataframe().slice(offset, length))
