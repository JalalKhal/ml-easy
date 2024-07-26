import pickle
from abc import abstractmethod
from typing import TypeVar, Generic

from recipes.steps.train.models import Model

class ModelRegistry:
    def __init__(self):
        pass

    @abstractmethod
    def log_model(self,
                  model: Model,
                  artifact_path: str) -> None:
        pass


class PickleModelRegistry(ModelRegistry):
    def __init__(self):
        super().__init__()

    def log_model(self,
                  model: Model,
                  artifact_path: str) -> None:
        with open(artifact_path, "wb") as f:
            pickle.dump(model, f)