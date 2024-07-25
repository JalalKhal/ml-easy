from recipes.interfaces.config import BaseRecipeConfig, BaseStepsConfig
from recipes.steps.steps_config import BaseIngestConfig, BaseTransformConfig, BaseSplitConfig, BaseTrainConfig, \
    BaseEvaluateConfig


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

class ClassificationStepsConfig(BaseStepsConfig):
    ingest: ClassificationIngestConfig
    transform: ClassificationTransformConfig
    split: ClassificationSplitConfig
    train: ClassificationTrainConfig
    evaluate: ClassificationEvaluateConfig


class ClassificationRecipeConfig(BaseRecipeConfig):
    steps: ClassificationStepsConfig



"""
    split: SplitConfig
    train: TrainConfig
    evaluate: Optional[EvaluateConfig] = None
    register_: RegisterConfig
    # Optional fields
    # ingest_scoring: Optional[str] = Field(alias='{{INGEST_SCORING_CONFIG}}')
    # predict: Optional[dict] = None

"""