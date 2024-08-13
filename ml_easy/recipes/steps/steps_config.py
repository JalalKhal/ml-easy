from abc import abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator

from ml_easy.recipes.enum import ScoreType, SourceType
from ml_easy.recipes.interfaces.config import BaseStepConfig


class RecipePathsConfig(BaseModel):
    recipe_root_path: str
    profile: Optional[str] = None


class SQLCredentialsConfig(BaseModel):
    username: str
    password: str
    hostname: str
    port: str
    database_name: str


class BaseIngestConfig(BaseStepConfig):
    ingest_fn: str
    table_name: str
    credentials: SQLCredentialsConfig


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


class SourceConfig(BaseModel):
    type: SourceType

    @property
    @abstractmethod
    def get_config(self) -> BaseModel:
        pass


class SqlConfig(BaseModel):
    hostname: str
    port: str
    user: str
    database_name: str
    table_name: str


class SqlAlchemyBasedSourceConfig(SourceConfig):
    config: SqlConfig

    @property
    def get_config(self) -> BaseModel:
        return self.config

    @field_validator('type')
    @classmethod
    def check_type(cls, type: SourceType):
        if type != SourceType.SQL_ALCHEMY_BASED:
            raise ValueError(f'Type must be equal to {SourceType.SQL_ALCHEMY_BASED} for {cls.__class__.__name__}')
        return type


class BaseRegisterConfig(BaseStepConfig):
    register_fn: str
    artifact_path: str
    registered_model_name: Optional[str]
    source: SqlAlchemyBasedSourceConfig
