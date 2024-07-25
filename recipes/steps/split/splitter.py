import numpy as np
from typing import List, Tuple

from recipes.enum import MLFlowErrorCode
from recipes.exceptions import MlflowException
from recipes.steps.ingest.datasets import Dataset


class DatasetSplitter:
    def __init__(self, val_prop: float, test_prop: float):
        self._val_prop = val_prop
        self._test_prop = test_prop
        self._train_prop = 1 - self._val_prop - self._test_prop

    def split(self, dataset: Dataset) -> Tuple[Dataset, Dataset, Dataset]:
        return dataset.split(self._train_prop, self._val_prop)
