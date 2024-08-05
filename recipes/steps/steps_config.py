from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from recipes.enum import ScoreType
from recipes.interfaces.config import BaseStepConfig


class RecipePathsConfig(BaseModel):
    recipe_root_path: str
    profile: Optional[str] = None


class BaseIngestConfig(BaseStepConfig):
    ingest_fn: str
    location: str
    sep: str
    encoding: str


class BaseSplitConfig(BaseStepConfig):
    split_fn: str
    split_ratios: List[float]


class BaseTransformConfig(BaseStepConfig):
    transformer_fn: str


class Score(BaseModel):
    name: ScoreType
    params: Dict[str, Any]


class BaseTrainConfig(BaseStepConfig):
    estimator_fn: str
    loss: str
    validation_metric: Score


class EvaluateCriteria(BaseStepConfig):
    metric: Score
    threshold: float


class BaseEvaluateConfig(BaseStepConfig):
    validation_criteria: List[EvaluateCriteria]


class BaseRegisterConfig(BaseStepConfig):
    register_fn: str
    artifact_path: str
    registered_model_name: Optional[str]
