import copy
import hashlib
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
from mlflow.data.dataset import Dataset as MLflowDataset
from polars._typing import ColumnNameOrSelector, ConcatMethod
from scipy.sparse import csr_matrix, hstack, vstack

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

    def __getitem__(self, indices):
        return self._getitem(indices)

    @abstractmethod
    def _getitem(self, indices):
        pass

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        pass

    @abstractmethod
    def to_pandas(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def to_numpy(self) -> np.ndarray[Any, Any]:
        pass

    @abstractmethod
    def to_csr(self) -> csr_matrix:
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

    @property
    @abstractmethod
    def columns(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def dtypes(self) -> List[str]:
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
    def slice(self, offset: int, length: int | None = None) -> Self:
        pass

    @abstractmethod
    def map_str(self, udf_map: Dict[str, Callable[[str], str]]) -> Self:
        pass

    def split(self, train_prop: float, val_prop: float) -> Tuple[List[int], List[int], List[int]]:
        total_samples = self.shape[0]
        train_size = int(train_prop * total_samples)
        val_size = int(val_prop * total_samples)

        indices = np.arange(total_samples)
        np.random.shuffle(indices)

        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[train_size + val_size :]

        return train_indices.tolist(), val_indices.tolist(), test_indices.tolist()

    @property
    @abstractmethod
    def hash_dataset(self) -> str:
        pass

    @abstractmethod
    def get_mlflow_dataset(self, source: str) -> MLflowDataset:
        pass


class PolarsDataset(Dataset[pl.DataFrame | pl.LazyFrame]):
    def __init__(self, service: pl.DataFrame | pl.LazyFrame):
        super().__init__(service)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.service.shape

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

    @property
    def columns(self) -> List[str]:
        return self.get_dataframe().columns

    @property
    def dtypes(self) -> List[str]:
        return [str(dtype) for dtype in self.service.dtypes]

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
        return self.__class__(self.service.slice(offset, length))

    def map_str(self, udf_map: Dict[str, Callable[[str], str]]) -> Self:
        maps = [pl.col(col).map_elements(udf_map[col], return_dtype=pl.Utf8) for col in udf_map]
        return self.__class__(service=self.service.with_columns_seq(maps))

    def to_csr(self) -> csr_matrix:
        return csr_matrix(self.to_numpy())

    def _getitem(self, indices):
        return self.__class__(self.get_dataframe().__getitem__(indices))

    @property
    def hash_dataset(self) -> str:
        row_hashes = self.service.hash_rows(seed=42)
        hasher = hashlib.sha256()
        for row_hash in row_hashes:
            hasher.update(row_hash.to_bytes(64, 'little'))
        return hasher.digest().hex()

    def get_mlflow_dataset(self, source: str) -> MLflowDataset:
        from mlflow.data.dataset_source_registry import resolve_dataset_source

        class PolarsMLFlowDataset(MLflowDataset):
            def __init__(self, ds: PolarsDataset):
                nonlocal source
                self.dataset = ds
                source = resolve_dataset_source(source)
                name = f"PolarsDataset_{self.dataset.shape[0]}x{self.dataset.shape[1]}"
                super().__init__(source=source, name=name)
                self.columns = self.dataset.columns
                self.df = self.dataset.get_dataframe()

            def _compute_digest(self) -> str:
                return self.dataset.hash_dataset

            def to_dict(self) -> Dict[str, str]:
                base_dict = super().to_dict()
                base_dict.update(
                    {
                        'shape': str(self.dataset.shape),
                        'columns': str(self.columns),
                        'dtypes': str({col: str(dtype) for col, dtype in zip(self.columns, self.dataset.dtypes)}),
                    }
                )
                return base_dict

            @property
            def profile(self) -> Optional[Any]:
                return {
                    'shape': self.dataset.shape,
                    'columns': self.dataset.columns,
                    'dtypes': {col: str(dtype) for col, dtype in zip(self.dataset.columns, self.dataset.dtypes)},
                    'null_count': self.df.null_count().to_dict(),
                    'memory_usage': self.df.estimated_size(),
                }

            @property
            def schema(self) -> Optional[Any]:
                return {col: str(dtype) for col, dtype in zip(self.dataset.columns, self.dataset.dtypes)}

        return PolarsMLFlowDataset(self)


class CsrMatrixDataset(Dataset[csr_matrix]):

    def __init__(self, service: csr_matrix):
        super().__init__(service)

    def __iter__(self) -> Iterable:
        coo = self.service.tocoo()
        for i, j, v in zip(coo.row, coo.col, coo.data):
            yield (i, j, v)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.service.shape

    def to_pandas(self) -> pd.DataFrame:
        coo = self.service.tocoo()
        df = pd.DataFrame({'row': coo.row, 'col': coo.col, 'data': coo.data})
        return df.pivot(index='row', columns='col', values='data').fillna(0)

    def to_numpy(self) -> np.ndarray:
        return self.service.toarray()

    @classmethod
    def from_numpy(cls, data: np.ndarray) -> Self:
        csr = csr_matrix(data)
        return cls(service=csr)

    def select(self, cols: List[int]) -> Self:
        sub_matrix = self.service[:, cols]
        return CsrMatrixDataset(sub_matrix)

    @property
    def columns(self) -> List[int]:
        return list(range(self.service.shape[1]))

    @property
    def dtypes(self) -> List[str]:
        return [str(self.service.dtype)]

    def collect(self) -> Self:
        return self

    def filter(self, filters: Dict[str, Optional[Union['EqualFilter[str]', 'InFilter[str]']]]) -> Self:
        raise NotImplementedError('Filtering not implemented for CSR matrices.')

    def drop_nulls(self, subset: Union[str, List[str], None] = None) -> Self:
        raise NotImplementedError('Drop nulls not implemented for CSR matrices.')

    @classmethod
    def concat(
        cls, items: Iterable[Self], *, how: str = 'vertical', rechunk: bool = False, parallel: bool = True
    ) -> Self:
        matrices = [item.service for item in items]
        if how == 'vertical':
            concatenated = vstack(matrices)
        elif how == 'horizontal':
            concatenated = hstack(matrices)
        else:
            raise ValueError(f"Invalid how argument: {how}")
        return cls(service=concatenated)

    def concatenate(
        self, items: Iterable[Self], *, how: str = 'vertical', rechunk: bool = False, parallel: bool = True
    ) -> Self:
        return self.__class__.concat([self] + list(items), how=how, rechunk=rechunk, parallel=parallel)

    def slice(self, offset: int, length: Union[int, None] = None) -> Self:
        if length is None:
            length = self.service.shape[0] - offset
        sub_matrix = self.service[offset : offset + length, :]
        return CsrMatrixDataset(sub_matrix)

    def map_str(self, udf_map: Dict[str, Callable[[str], str]]) -> Self:
        raise NotImplementedError('String mapping not implemented for CSR matrices.')

    def to_csr(self) -> csr_matrix:
        return copy.deepcopy(self.service)

    def _getitem(self, indices):
        return self.__class__(self.service.__getitem__(indices))

    @property
    def hash_dataset(self) -> str:
        m = hashlib.sha256()
        m.update(self.service.data.tobytes())
        return m.hexdigest()

    def get_mlflow_dataset(self, source: str) -> MLflowDataset:
        from mlflow.data.dataset_source_registry import resolve_dataset_source

        class CsrMatrixMLFlowDataset(MLflowDataset):
            def __init__(self, ds: CsrMatrixDataset):
                nonlocal source
                self.dataset = ds
                source = resolve_dataset_source(source)
                name = f"CsrMatrix_{self.dataset.shape[0]}x{self.dataset.shape[1]}"
                super().__init__(source, name)

            def _compute_digest(self) -> str:
                return self.dataset.hash_dataset

            def to_dict(self) -> Dict[str, str]:
                base_dict = super().to_dict()
                base_dict.update(
                    {
                        'shape': str(self.dataset.shape),
                        'nnz': str(self.dataset.service.nnz),
                        'dtype': str(self.dataset.service.dtype),
                    }
                )
                return base_dict

            @property
            def profile(self) -> Optional[Any]:
                return {
                    'shape': self.dataset.shape,
                    'nnz': self.dataset.service.nnz,
                    'density': self.dataset.service.nnz / (self.dataset.shape[0] * self.dataset.shape[1]),
                    'dtype': str(self.dataset.service.dtype),
                }

            @property
            def schema(self) -> Optional[Any]:
                return None

        return CsrMatrixMLFlowDataset(self)
