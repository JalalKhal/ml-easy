from recipes.classification.v1.config import ClassificationRegisterConfig
from recipes.interfaces.config import Context
from recipes.steps.register.ModelRegistry import ModelRegistry, PickleModelRegistry


def register_fn(conf: ClassificationRegisterConfig, context: Context) -> ModelRegistry:
    return PickleModelRegistry()
