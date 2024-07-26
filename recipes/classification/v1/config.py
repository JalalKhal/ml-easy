from recipes.interfaces.config import BaseRecipeConfig, BaseStepsConfig
from recipes.steps.steps_config import BaseIngestConfig, BaseTransformConfig, BaseSplitConfig, BaseTrainConfig, \
    BaseEvaluateConfig, BaseRegisterConfig


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
    steps: ClassificationStepsConfig


