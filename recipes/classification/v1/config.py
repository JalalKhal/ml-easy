from recipes.interfaces.config import BaseRecipeConfig, BaseStepsConfig
from recipes.steps.steps_config import (
    BaseEvaluateConfig,
    BaseIngestConfig,
    BaseRegisterConfig,
    BaseSplitConfig,
    BaseTrainConfig,
    BaseTransformConfig,
)


class ClassificationIngestConfig(BaseIngestConfig):
    pass


class ClassificationTransformConfig(BaseTransformConfig):
    pass


class ClassificationSplitConfig(BaseSplitConfig):
    pass


class ClassificationTrainConfig(BaseTrainConfig):
    pass


class ClassificationEvaluateConfig(BaseEvaluateConfig):
    pass


class ClassificationRegisterConfig(BaseRegisterConfig):
    pass


class ClassificationStepsConfig(BaseStepsConfig):
    ingest: ClassificationIngestConfig
    transform: ClassificationTransformConfig
    split: ClassificationSplitConfig
    train: ClassificationTrainConfig
    evaluate: ClassificationEvaluateConfig
    register_: ClassificationRegisterConfig


class ClassificationRecipeConfig(BaseRecipeConfig):
    pass
