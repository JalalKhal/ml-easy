import copy
import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    IO,
    Any,
    Callable,
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
from mlflow.data import DatasetSource  # type: ignore
from mlflow.data.dataset import Dataset as MLflowDataset  # type: ignore
from mlflow.types.utils import _infer_schema  # type: ignore
from polars._typing import ConcatMethod, IntoExpr, SchemaDict
from scipy.sparse import csr_matrix, hstack, vstack  # type: ignore
from sqlalchemy import create_engine

from ml_easy.recipes.enum import MLFlowErrorCode
from ml_easy.recipes.exceptions import MlflowException
from ml_easy.recipes.steps.steps_config import SourceConfig
from ml_easy.recipes.steps.transform.filters import EqualFilter, InFilter

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
    def select(self, *args: Any, **kwargs: Any) -> Self:
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
    def filter(self, filters: Dict[str, List[Union[EqualFilter[str], InFilter[str]]]]) -> Self:
        pass

    @abstractmethod
    def drop_nulls(
        self,
        subset: Union[str, List[str], None] = None,
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
    def get_mlflow_dataset(self, conf: SourceConfig) -> MLflowDataset:
        pass


class PolarsDataset(Dataset[pl.DataFrame | pl.LazyFrame]):
    def __init__(self, service: pl.DataFrame | pl.LazyFrame):
        super().__init__(service)
        self._get_dataframe: Optional[pl.DataFrame] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.get_dataframe.shape

    def to_pandas(self):
        return self.get_dataframe.to_pandas()

    @property
    def get_dataframe(self) -> pl.DataFrame:
        if self._get_dataframe is None:
            self._get_dataframe = self.service.collect() if isinstance(self.service, pl.LazyFrame) else self.service
        return self._get_dataframe

    def __iter__(self) -> Iterable:
        ds: pl.DataFrame = self.get_dataframe
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
        return self.get_dataframe.to_numpy()

    @classmethod
    def from_numpy(
        cls,
        data: np.ndarray[Any, Any],
    ) -> Self:
        return cls(pl.from_numpy(data))

    @classmethod
    def from_pandas(
        cls,
        data: pd.DataFrame,
        *,
        schema_overrides: SchemaDict | None = None,
        rechunk: bool = True,
        nan_to_null: bool = True,
        include_index: bool = False,
    ) -> Self:
        return cls(
            pl.from_pandas(
                data,
                schema_overrides=schema_overrides,
                rechunk=rechunk,
                nan_to_null=nan_to_null,
                include_index=include_index,
            )
        )

    @classmethod
    def from_sql_database(cls, table_name: str, credentials: Dict[str, str]) -> Self:
        username = credentials['username']
        password = credentials['password']
        hostname = credentials['hostname']
        database_name = credentials['database_name']
        port = credentials['port']
        connection_string = f'postgresql+psycopg2://{username}:{password}@{hostname}:{port}/{database_name}'
        engine = create_engine(connection_string)
        query = f"SELECT * FROM {table_name}"
        return cls.from_pandas(pd.read_sql(query, engine))

    @classmethod
    def concat(
        cls, items: Iterable[Self], *, how: ConcatMethod = 'vertical', rechunk: bool = False, parallel: bool = True
    ) -> Self:
        return cls(pl.concat([it.service for it in items], how=how, rechunk=rechunk, parallel=parallel))  # type: ignore

    def concatenate(
        self, items: Iterable[Self], *, how: ConcatMethod = 'vertical', rechunk: bool = False, parallel: bool = True
    ) -> Self:
        return self.__class__.concat([self] + list(items), how='horizontal', rechunk=rechunk, parallel=parallel)

    def select(self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr) -> Self:
        return self.__class__(service=self.service.select(*exprs, **named_exprs))

    @property
    def columns(self) -> List[str]:
        if isinstance(self.service, pl.LazyFrame):
            return self.service.collect_schema().names()
        return self.service.columns

    @property
    def dtypes(self) -> List[str]:
        dtypes = (
            self.service.dtypes if isinstance(self.service, pl.DataFrame) else self.service.collect_schema().dtypes()
        )
        return [str(dtype) for dtype in dtypes]

    def collect(self) -> Self:
        return self.__class__(self.get_dataframe)

    def drop_nulls(
        self,
        subset: Union[str, List[str], None] = None,
    ) -> Self:
        return self.__class__(self.service.drop_nulls(subset))

    def filter(self, filters: Dict[str, List[Union[EqualFilter[str], InFilter[str]]]]) -> Self:
        from ml_easy.recipes.utils import is_instance_for_generic

        def _get_expr_filter(col_filter: Union[EqualFilter[str], InFilter[str]]):

            if is_instance_for_generic(col_filter, EqualFilter[str]):
                return (
                    ~(pl.col(col) == col_filter.value) if col_filter.neg else pl.col(col) == col_filter.value
                )  # type:ignore
            elif is_instance_for_generic(col_filter, InFilter[str]):
                return (
                    ~(pl.col(col).is_in(col_filter.values)) if col_filter.neg else pl.col(col).is_in(col_filter.values)
                )  # type:ignore
            else:
                raise MlflowException(
                    message=f'Unsupported filter type {col_filter.__class__.__name__}',
                    error_code=MLFlowErrorCode.INVALID_PARAMETER_VALUE,
                )

        predicates_expr = []
        for col in filters:
            col_filters = filters[col]
            for filter in col_filters:  # type: ignore
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
        return self.__class__(self.get_dataframe.__getitem__(indices))

    @property
    def hash_dataset(self) -> str:
        row_hashes = self.get_dataframe.hash_rows(seed=42)
        hasher = hashlib.sha256()
        for row_hash in row_hashes:
            hasher.update(row_hash.to_bytes(64, 'little'))
        return hasher.digest().hex()

    def get_mlflow_dataset(self, conf: SourceConfig) -> MLflowDataset:
        from ml_easy.recipes.utils import resolve_dataset_source

        class PolarsMLFlowDataset(MLflowDataset):
            def __init__(self, ds: PolarsDataset):
                nonlocal conf
                self.dataset = ds
                source: DatasetSource = resolve_dataset_source(conf)
                name = f"PolarsDataset_{self.dataset.shape[0]}x{self.dataset.shape[1]}"
                super().__init__(source=source, name=name)
                self.columns = self.dataset.columns

            def _compute_digest(self) -> str:
                return self.dataset.hash_dataset

            def to_dict(self) -> Dict[str, str]:
                base_dict = super().to_dict()
                base_dict.update(
                    {
                        'shape': str(self.dataset.shape),
                        'columns': str(self.columns),
                        'dtypes': str({col: str(dtype) for col, dtype in zip(self.columns, self.dataset.dtypes)}),
                        'schema': str(self.schema),
                        'profile': str(self.profile),
                    }
                )
                return base_dict

            @property
            def profile(self) -> Optional[Any]:
                return {
                    'shape': self.dataset.shape,
                    'columns': self.dataset.columns,
                    'dtypes': {col: str(dtype) for col, dtype in zip(self.dataset.columns, self.dataset.dtypes)},
                }

            @property
            def schema(self) -> Optional[Any]:
                return _infer_schema(self.dataset.to_pandas())

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
        return self.__class__(sub_matrix)

    @property
    def columns(self) -> List[str]:
        return [str(i) for i in range(self.service.shape[1])]

    @property
    def dtypes(self) -> List[str]:
        return [str(self.service.dtype)]

    def collect(self) -> Self:
        return self

    def filter(self, filters: Dict[str, List[Union['EqualFilter[str]', 'InFilter[str]']]]) -> Self:
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
        return self.__class__(sub_matrix)

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

    def get_mlflow_dataset(self, conf: SourceConfig) -> MLflowDataset:
        from ml_easy.recipes.utils import resolve_dataset_source

        class CsrMatrixMLFlowDataset(MLflowDataset):
            def __init__(self, ds: CsrMatrixDataset):
                nonlocal conf
                self.dataset = ds
                source = resolve_dataset_source(conf)
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
                        'schema': str(self.schema),
                        'profile': str(self.profile),
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
                return _infer_schema(self.dataset.to_numpy())

        return CsrMatrixMLFlowDataset(self)
