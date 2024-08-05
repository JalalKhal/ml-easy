from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, field_validator

from recipes.enum import FilterType
from recipes.interfaces.config import BaseRecipeConfig, BaseStepsConfig
from recipes.steps.steps_config import (
    BaseEvaluateConfig,
    BaseIngestConfig,
    BaseRegisterConfig,
    BaseSplitConfig,
    BaseTrainConfig,
    BaseTransformConfig,
)


class FilterConfig(BaseModel):
    type: FilterType
    values: Union[str, List[str]]


class ClassificationIngestConfig(BaseIngestConfig):
    filters: Optional[Dict[str, FilterConfig]]


class LibraryEmbedder(BaseModel):
    path: str
    params: Dict[str, Any]

    @field_validator('params', mode='before')
    def check_scikit(cls, v):
        if 'ngram_range' in v:
            v['ngram_range'] = eval(v['ngram_range'])
        return v


class ClassificationTransformConfig(BaseTransformConfig):
    cols: Dict[str, LibraryEmbedder]


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
