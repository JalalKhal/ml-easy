from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, field_validator

from ml_easy.recipes.enum import FilterType
from ml_easy.recipes.interfaces.config import BaseRecipeConfig, BaseStepsConfig
from ml_easy.recipes.steps.steps_config import (
    BaseEvaluateConfig,
    BaseIngestConfig,
    BaseRegisterConfig,
    BaseSplitConfig,
    BaseTrainConfig,
    BaseTransformConfig,
)
from ml_easy.recipes.steps.transform.formatter.formatter import TextFormatterConfig


class FilterConfig(BaseModel):
    type: FilterType


class EqualFilterConfig(FilterConfig):
    neg: bool
    value: str

    @field_validator('type')
    @classmethod
    def check_type(cls, type: FilterType):
        if type != FilterType.EQUAL:
            raise ValueError('Type must be EqualFilter for EqualFilterConfig')
        return type


class InFilterConfig(FilterConfig):
    neg: bool
    values: List[str]

    @field_validator('type')
    @classmethod
    def check_type(cls, type: FilterType):
        if type != FilterType.IN:
            raise ValueError('Type must be InFilter for InFilterConfig')
        return type


class ClassificationIngestConfig(BaseIngestConfig):
    pass


class LibraryEmbedder(BaseModel):
    path: str
    params: Dict[str, Any]

    @field_validator('params', mode='before')
    def check_scikit(cls, v):
        if 'ngram_range' in v:
            v['ngram_range'] = eval(v['ngram_range'])
        return v


class ColConfig(BaseModel):
    embedder: Optional[LibraryEmbedder] = None
    formatter: Optional[TextFormatterConfig] = None
    filters: Optional[List[Union[EqualFilterConfig, InFilterConfig]]] = None


class ClassificationTransformConfig(BaseTransformConfig):
    cols: Dict[str, ColConfig]


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

    @property
    def get_steps(self) -> BaseStepsConfig:
        return self.steps
