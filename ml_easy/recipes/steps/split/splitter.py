from typing import Tuple

from ml_easy.recipes._typing import TupleDataset
from ml_easy.recipes.steps.ingest.datasets import Dataset


class DatasetSplitter:
    def __init__(self, val_prop: float, test_prop: float):
        self._val_prop = val_prop
        self._test_prop = test_prop
        self._train_prop = 1 - self._val_prop - self._test_prop

    def split(self, X: Dataset, y: Dataset) -> Tuple[TupleDataset, TupleDataset, TupleDataset]:
        train_indices, val_indices, test_indices = y.split(self._train_prop, self._val_prop)
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
