from recipes.classification.v1.config import ClassificationRegisterConfig
from recipes.interfaces.config import Context
from recipes.steps.register.registry import MlflowRegistry, Registry


def register_fn(conf: ClassificationRegisterConfig, context: Context) -> Registry:
    return MlflowRegistry(conf, context)
