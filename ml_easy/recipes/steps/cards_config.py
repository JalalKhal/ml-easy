from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict

from ml_easy.recipes._typing import TupleDataset
from ml_easy.recipes.interfaces.config import BaseCard
from ml_easy.recipes.steps.ingest.datasets import Dataset
from ml_easy.recipes.steps.steps_config import BaseTransformConfig, Score
from ml_easy.recipes.steps.train.models import Model


class IngestCard(BaseCard):
    dataset: Optional[Dataset] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class TransformCard(BaseCard):
    tf_dataset: Optional[TupleDataset] = None
    config: Optional[BaseTransformConfig] = None
    transformer_path: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class SplitCard(BaseCard):
    train_val_test: Optional[Tuple[TupleDataset, TupleDataset, TupleDataset]] = None
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


class RegisterCard(BaseCard):
    pass


class StepMessage(BaseModel):
    ingest: Optional[IngestCard] = None
    transform: Optional[TransformCard] = None
    split: Optional[SplitCard] = None
    train: Optional[TrainCard] = None
    evaluate: Optional[EvaluateCard] = None
    register_: Optional[RegisterCard] = None
