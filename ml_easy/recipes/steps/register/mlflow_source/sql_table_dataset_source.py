from abc import ABC
from typing import Any, Dict, Self, Type

from mlflow.data import DatasetSource  # type: ignore


class DatasetSourceWrapper(ABC, DatasetSource):
    @classmethod
    def load_from_path(cls, path: str) -> Type[Self]:
        from ml_easy.recipes.utils import get_class_from_string

        return get_class_from_string(path)


class SQLTableDatasetSource(DatasetSourceWrapper):
    def __init__(self, hostname: str, port: str, user: str, database_name: str, table_name: str):
        self._hostname = hostname
        self._port = port
        self._user = user
        self._database_name = database_name
        self._table_name = table_name

    @staticmethod
    def _get_source_type() -> str:
        return 'sql-alchemy_based'

    def load(self, **kwargs):
        """
        Load is not implemented for Code Dataset Source.
        """
        raise NotImplementedError

    @staticmethod
    def _can_resolve(raw_source: Any):
        return False

    @classmethod
    def _resolve(cls, raw_source: str) -> 'SQLTableDatasetSource':
        raise NotImplementedError

    def to_dict(self) -> Dict[Any, Any]:
        return {
            'hostname': self._hostname,
            'port': self._port,
            'user': self._user,
            'database_name': self._database_name,
            'table_name': self._table_name,
        }

    @classmethod
    def from_dict(cls, source_dict: Dict[Any, Any]) -> 'SQLTableDatasetSource':
        return cls(**source_dict)
