from abc import ABC, abstractmethod
from typing import Any, List, Type, TypeVar

from typing_extensions import Generic

U = TypeVar('U')


class Filter(ABC, Generic[U]):
    def __init__(self, neg: bool):
        self.neg = neg

    @abstractmethod
    def filter(self, x: U) -> bool:
        pass

    @classmethod
    def load_from_path(cls, path: str) -> Any:
        from ml_easy.recipes.utils import get_class_from_string

        filter_class: Type[Filter] = get_class_from_string(path)
        return filter_class


class EqualFilter(Filter[U], Generic[U]):
    def __init__(self, value: U, neg: bool):
        super().__init__(neg)
        self.value = value

    def filter(self, x: U) -> bool:
        pos = x == self.value
        return not pos if self.neg else pos


class InFilter(Filter[U], Generic[U]):
    def __init__(self, values: List[U], neg: bool):
        super().__init__(neg)
        self.values = values

    def filter(self, x: U) -> bool:
        pos = x in self.values
        return not pos if self.neg else pos
