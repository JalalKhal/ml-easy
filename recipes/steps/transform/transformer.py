import importlib
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Protocol, Self, TypeVar, Union

from recipes.classification.v1.config import ClassificationTransformConfig
from recipes.interfaces.config import Context
from recipes.steps.ingest.datasets import CsrMatrixDataset, Dataset
from recipes.steps.transform.filters import EqualFilter, InFilter
from recipes.steps.transform.formatter.formatter import AvsCleaner, AvsLemmatizer

U = TypeVar('U')


class Transformer(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X: Dataset) -> None:
        pass

    @abstractmethod
    def transform(self, X: Dataset) -> Dataset:
        pass

    def fit_transform(self, X: Dataset) -> Dataset:
        self.fit(X)
        return self.transform(X)


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

    def fit(self, X: Dataset) -> None:
        self._service.fit(X.to_numpy().reshape(-1))

    def transform(self, X: Dataset) -> Dataset:
        ds_tf = self._service.transform(X.to_numpy().reshape(-1))
        return CsrMatrixDataset(ds_tf)


class MultipleTfIdfTransformer(Transformer):
    def __init__(self, conf: ClassificationTransformConfig, context: Context):
        super().__init__()
        self.conf = conf
        self.context = context
        self.embedder = {
            col: ScikitEmbedder.load_from_library(conf.cols[col].embedder.path, conf.cols[col].embedder.params)
            for col in conf.cols
            if conf.cols[col].embedder
        }

    def fit(self, X: Dataset) -> None:
        for col in self.embedder:
            self.embedder[col].fit(X.select([col]))

    def transform(self, X: Dataset) -> Dataset:
        tfs_X: List[CsrMatrixDataset] = [
            CsrMatrixDataset(self.embedder[col].transform(X.select([col])).to_csr())
            for col in self.conf.cols
            if self.conf.cols[col].embedder
        ]
        return CsrMatrixDataset.concat(tfs_X, how='horizontal')


class FilterTransformer(Transformer):
    def __init__(self, filters: Dict[str, Optional[List[Union[EqualFilter[str], InFilter[str]]]]]):
        super().__init__()
        self.filters = filters

    def fit(self, X: Dataset) -> None:
        pass

    def transform(self, X: Dataset) -> Dataset:
        return X.filter(self.filters)


class FormaterTransformer(Transformer):

    def __init__(self, config: ClassificationTransformConfig):
        super().__init__()
        self.config = config
        self.text_cleaner = {
            col: AvsCleaner(config_settings=self.config.cols[col].formatter.cleaner)
            for col in self.config.cols
            if self.config.cols[col].formatter
        }
        self.lemmatizer = AvsLemmatizer()

    def fit(self, X: Dataset) -> None:
        pass

    def transform(self, X: Dataset) -> Dataset:
        # TO DO : Can be optimized take too long time to process
        def func(col):
            def _func(x):
                return self.lemmatizer(self.text_cleaner[col](x))

            return _func

        return X.map_str({col: func(col) for col in self.config.cols if self.config.cols[col].formatter})


class PipelineTransformer(Transformer):
    def __init__(self, transformers: List[Transformer]):
        super().__init__()
        self._transformers = transformers

    def fit(self, X: Dataset) -> None:
        for transformer in self._transformers[:-1]:
            X = transformer.fit_transform(X)
        self._transformers[-1].fit(X)

    def transform(self, X: Dataset) -> Dataset:
        for transformer in self._transformers:
            X = transformer.transform(X)
        return X
