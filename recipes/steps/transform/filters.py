from abc import ABC, abstractmethod
from typing import Any, List, Type, TypeVar

from typing_extensions import Generic

U = TypeVar('U')


class Filter(ABC, Generic[U]):
    def __init__(self):
        pass

    @abstractmethod
    def filter(self, x: U) -> bool:
        pass

    @classmethod
    def load_from_path(cls, path: str) -> Any:
        from recipes.utils import get_class_from_string

        filter_class: Type[Filter] = get_class_from_string(path)
        return filter_class


class EqualFilter(Filter[U], Generic[U]):
    def __init__(self, value: U):
        super().__init__()
        self.value = value

    def filter(self, x: U) -> bool:
        return x == self.value


class InFilter(Filter[U], Generic[U]):
    def __init__(self, values: List[U]):
        super().__init__()
        self.values = values

    def filter(self, x: U) -> bool:
        return x in self.values
