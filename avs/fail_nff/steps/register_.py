from ml_easy.recipes.classification.v1.config import ClassificationRegisterConfig
from ml_easy.recipes.interfaces.config import Context
from ml_easy.recipes.steps.register.registry import MlflowRegistry, Registry


def register_fn(conf: ClassificationRegisterConfig, context: Context) -> Registry:
    return MlflowRegistry(conf, context)
