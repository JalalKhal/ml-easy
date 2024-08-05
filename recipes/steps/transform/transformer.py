import importlib
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Protocol, Self, TypeVar

from numpy import hstack

from recipes.classification.v1.config import ClassificationTransformConfig
from recipes.interfaces.config import Context
from recipes.steps.ingest.datasets import Dataset, PolarsDataset

U = TypeVar('U')


class Transformer(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X: Dataset, y: Dataset = None) -> None:
        pass

    @abstractmethod
    def transform(self, X: Dataset, y: Dataset = None) -> Dataset:
        pass

    def fit_transform(self, X: Dataset, y: Dataset = None) -> Dataset:
        self.fit(X, y)
        return self.transform(X, y)


class LibraryTransformer(Transformer, ABC, Generic[U]):

    def __init__(self, service: U):
        super().__init__()
        self._service = service

    @classmethod
    def load_from_library(cls, path: str, params: Dict[str, Any]) -> Self:
        module_path, class_name = path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        protocol_methods = [method for method in Transformer.__annotations__.keys()]
        if not all(hasattr(model_class, method) for method in protocol_methods):
            raise ValueError(f"scikit-learn {class_name} estimator is not a {Transformer}")
        return cls(model_class(**params))


class ScikitService(Protocol):
    def fit(self, raw_documents, y=None): ...

    def transform(self, raw_documents): ...


class ScikitEmbedder(LibraryTransformer):
    def __init__(self, service: ScikitService):
        super().__init__(service)

    def fit(self, X: Dataset, y: Dataset = None) -> None:
        self._service.fit(X.to_numpy().reshape(-1))

    def transform(self, X: Dataset, y: Dataset = None) -> Dataset:
        ds_tf = self._service.transform(X.to_numpy().reshape(-1))
        return PolarsDataset.from_numpy(data=ds_tf.toarray())


class MultipleTfIdfTransformer(Transformer):
    def __init__(self, conf: ClassificationTransformConfig, context: Context):
        super().__init__()
        self.conf = conf
        self.context = context
        self.embedder = {
            col: ScikitEmbedder.load_from_library(conf.cols[col].path, conf.cols[col].params) for col in conf.cols
        }

    def fit(self, X: Dataset, y: Dataset = None) -> None:
        for col in self.conf.cols:
            self.embedder[col].fit(X.select([col]), y)

    def transform(self, X: Dataset, y: Dataset = None) -> Dataset:
        X: List[PolarsDataset] = [
            PolarsDataset.from_numpy(
                hstack([self.embedder[col].transform(X.select([col]), y).to_numpy() for col in self.conf.cols])
            ),
        ]
        if y:
            return PolarsDataset.concat(X + [y], how='horizontal')
        else:
            return PolarsDataset.concat(X, how='horizontal')
