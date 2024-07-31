import importlib
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Protocol, Self, Dict, Any, Type

import numpy as np
from sklearn.base import is_classifier # type: ignore

from recipes.steps.evaluate.score import Score
from recipes.steps.ingest.datasets import Dataset

U = TypeVar('U')


class Model(ABC, Generic[U]):
    def __init__(self, service: U):
        self._service = service

    @abstractmethod
    def fit(self, X: Dataset, y: Dataset) -> None:
        pass

    @abstractmethod
    def predict(self, X: Dataset) -> Dataset:
        pass

    def fit_predict(self, X: Dataset, y: Dataset) -> Dataset:
        self.fit(X, y)
        return self.predict(X)

    @abstractmethod
    def score(self,
              X: Dataset,
              y: Dataset,
              metric: Type[Score],
              **kwargs
              ) -> float:
        pass

    @abstractmethod
    def get_model_outputs(self) -> Dict[str, Any]:
        pass




class EstimatorProtocol(Protocol):
    def fit(self, X, y, sample_weight=None):
        ...

    def predict(self, X):
        ...

    def predict_proba(self, X):
        ...

    def get_params(self, deep=True):
        ...

class ScikitModel(Model[EstimatorProtocol]):
    def __init__(self, service: EstimatorProtocol):
        super().__init__(service)

    def fit(self, X: Dataset, y: Dataset) -> None:
        self._service.fit(X.to_numpy(), y.to_numpy().reshape(-1))

    def predict(self, X: Dataset) -> Dataset:
        y: np.ndarray = self._service.predict(X.to_numpy())
        return X.from_numpy(y)

    @classmethod
    def load_from_library(cls, path: str, params: Dict[str, Any]) -> Self:
        module_path, class_name = path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        protocol_methods = [method for method in EstimatorProtocol.__annotations__.keys()]
        if not all(hasattr(model_class, method) for method in protocol_methods):
            raise ValueError(f"scikit-learn {class_name} estimator is not a {EstimatorProtocol}")
        return cls(model_class(**params))

    def score(self,
              X: Dataset,
              y: Dataset,
              metric: Type[Score],
              **kwargs
              ) -> float:
        return metric.score(y, self.predict(X), **kwargs)

    def get_model_outputs(self) -> Dict[str, Any]:
        outputs = {}
        outputs['model_type'] = type(self._service).__name__
        outputs['is_classifier'] = is_classifier(self._service)
        outputs['params'] = self._service.get_params()
        return outputs
