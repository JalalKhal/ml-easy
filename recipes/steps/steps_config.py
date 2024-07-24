from typing import List, Optional, Union
from pydantic import BaseModel, Field
from recipes.interfaces.cards_config import BaseStepConfig, BaseRecipeConfig


class IngestConfig(BaseStepConfig):
    using: str


class SplitConfig(BaseStepConfig):
    split_ratios: List[float] = Field(default=[0.75, 0.125, 0.125])
    post_split_filter_method: Optional[str] = Field(default="create_dataset_filter")


class TransformConfig(BaseStepConfig):
    using: str = Field(default="custom")
    transformer_method: Optional[str] = Field(default="transformer_fn")


class TrainConfig(BaseStepConfig):
    using: str


class EvaluateCriteria(BaseStepConfig):
    metric: str
    threshold: Union[int, float]


class EvaluateConfig(BaseStepConfig):
    validation_criteria: Optional[List[EvaluateCriteria]] = None


class RegisterConfig(BaseStepConfig):
    allow_non_validated_model: bool = Field(default=False)


class StepsConfig(BaseModel):
    ingest: IngestConfig
    split: SplitConfig
    transform: TransformConfig
    train: TrainConfig
    evaluate: Optional[EvaluateConfig] = None
    register_: RegisterConfig
    # Optional fields
    # ingest_scoring: Optional[str] = Field(alias='{{INGEST_SCORING_CONFIG}}')
    # predict: Optional[dict] = None


class ClassificationRecipeConfig(BaseRecipeConfig):
    target_col: str
    primary_metric: str
    steps: StepsConfig
    # Optional fields
    # custom_metrics: Optional[List[dict]] = None
