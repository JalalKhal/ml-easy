from typing import Optional, List, Union

from pydantic import BaseModel, Field

from recipes.enum import Framework
from recipes.interfaces.config import BaseStepConfig


class RecipePathsConfig(BaseModel):
    recipe_root_path: str
    profile: Optional[str] = None


class BaseIngestConfig(BaseStepConfig):
    location: str
    framework: Framework


class BaseSplitConfig(BaseStepConfig):
    split_ratios: List[float] = Field(default=[0.75, 0.125, 0.125])
    post_split_filter_method: Optional[str] = Field(default="create_dataset_filter")


class BaseTransformConfig(BaseStepConfig):
    using: str = Field(default="custom")
    transformer_method: Optional[str] = Field(default="transformer_fn")


class BaseTrainConfig(BaseStepConfig):
    using: str


class EvaluateCriteria(BaseStepConfig):
    metric: str
    threshold: Union[int, float]


class EvaluateConfig(BaseStepConfig):
    validation_criteria: Optional[List[EvaluateCriteria]] = None


class RegisterConfig(BaseStepConfig):
    allow_non_validated_model: bool = Field(default=False)






