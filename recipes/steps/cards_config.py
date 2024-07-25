from typing import Optional, Tuple, Dict, Any, List

from pydantic import BaseModel, ConfigDict

from recipes.enum import ScoreType
from recipes.interfaces.config import BaseCard
from recipes.steps.ingest.datasets import Dataset
from recipes.steps.steps_config import Score
from recipes.steps.train.models import Model


class IngestCard(BaseCard):
    dataset: Optional[Dataset] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class TransformCard(BaseCard):
    tf_dataset: Optional[Dataset] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class SplitCard(BaseCard):
    train_val_test: Optional[Tuple[Dataset, Dataset, Dataset]] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class Metric(BaseModel):
    name: Score
    value: float


class TrainCard(BaseCard):
    mod: Optional[Model] = None
    mod_outputs: Optional[Dict[str, Any]] = None
    val_metric: Optional[float] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class EvaluateCard(BaseCard):
    metrics_eval: Optional[List[Metric]] = None


class StepMessage(BaseModel):
    ingest: Optional[IngestCard] = None
    transform: Optional[TransformCard] = None
    split: Optional[SplitCard] = None
    train: Optional[TrainCard] = None
    evaluate: Optional[EvaluateCard] = None
