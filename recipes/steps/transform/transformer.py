import abc

from recipes.steps.ingest.datasets import Dataset


class Transformer:

    def __init__(self):
        pass

    @abc.abstractmethod
    def fit(self, X: Dataset) -> None:
        pass

    @abc.abstractmethod
    def transform(self, X: Dataset) -> Dataset:
        pass
    def fit_transform(self, X: Dataset) -> Dataset:
        self.fit(X)
        return self.transform(X)