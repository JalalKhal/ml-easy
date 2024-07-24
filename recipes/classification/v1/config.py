from recipes.interfaces.config import BaseRecipeConfig, BaseStepsConfig
from recipes.steps.steps_config import BaseIngestConfig


class ClassificationIngestConfig(BaseIngestConfig):
    pass

class ClassificationStepsConfig(BaseStepsConfig):
    ingest: ClassificationIngestConfig


class ClassificationRecipeConfig(BaseRecipeConfig):
    target_col: str
    primary_metric: str
    steps: ClassificationStepsConfig
    # Optional fields
    # custom_metrics: Optional[List[dict]] = None


"""
    split: SplitConfig
    transform: TransformConfig
    train: TrainConfig
    evaluate: Optional[EvaluateConfig] = None
    register_: RegisterConfig
    # Optional fields
    # ingest_scoring: Optional[str] = Field(alias='{{INGEST_SCORING_CONFIG}}')
    # predict: Optional[dict] = None

"""