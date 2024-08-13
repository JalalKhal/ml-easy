import pickle
from typing import Any, List

from ml_easy.recipes.classification.v1.config import (
    ClassificationEvaluateConfig,
    ClassificationIngestConfig,
    ClassificationRegisterConfig,
    ClassificationSplitConfig,
    ClassificationTrainConfig,
    ClassificationTransformConfig,
)
from ml_easy.recipes.interfaces.config import Context
from ml_easy.recipes.steps.cards_config import Metric, StepMessage
from ml_easy.recipes.steps.evaluate.evaluate import EvaluateStep
from ml_easy.recipes.steps.ingest.datasets import Dataset
from ml_easy.recipes.steps.ingest.ingest import IngestStep
from ml_easy.recipes.steps.register.register_ import RegisterStep
from ml_easy.recipes.steps.register.registry import Registry
from ml_easy.recipes.steps.split.split import SplitStep
from ml_easy.recipes.steps.split.splitter import DatasetSplitter
from ml_easy.recipes.steps.train.models import Model
from ml_easy.recipes.steps.train.train import TrainStep
from ml_easy.recipes.steps.transform.transform import TransformStep
from ml_easy.recipes.utils import get_features_target, get_score_class


class ClassificationIngestStep(IngestStep[ClassificationIngestConfig]):
    def __init__(self, ingest_config: ClassificationIngestConfig, context: Context):
        super().__init__(ingest_config, context)

    def _run(self, message: StepMessage) -> StepMessage:
        dataset: Any = self.get_step_result()
        self.validate_step_result(dataset, Dataset)
        self.card.dataset = dataset
        return message


class ClassificationTransformStep(TransformStep[ClassificationTransformConfig]):
    def __init__(self, transform_config: ClassificationTransformConfig, context: Context):
        super().__init__(transform_config, context)

    def _run(self, message: StepMessage) -> StepMessage:
        from ml_easy.recipes.steps.transform.transformer import Transformer

        transformer: Any = self.get_step_result()
        self.validate_step_result(transformer, Transformer)
        self.card.transformer_path = f"{self.card.step_output_path}/transformer.pkl"
        X, y = get_features_target(message.ingest.dataset, self.context.target_col)  # type:ignore
        self.card.tf_dataset = (transformer.fit_transform(X), y)
        with open(self.card.transformer_path, 'wb') as f:
            pickle.dump(transformer, f)
        self.card.config = self.conf
        return message


class ClassificationSplitStep(SplitStep[ClassificationSplitConfig]):
    def __init__(self, split_config: ClassificationSplitConfig, context: Context):
        super().__init__(split_config, context)

    def _run(self, message: StepMessage) -> StepMessage:
        dataset_splitter: Any = self.get_step_result()
        self.validate_step_result(dataset_splitter, DatasetSplitter)
        self.card.train_val_test = dataset_splitter.split(*message.transform.tf_dataset)  # type: ignore
        return message


class ClassificationTrainStep(TrainStep[ClassificationTrainConfig]):
    def __init__(self, train_config: ClassificationTrainConfig, context: Context):
        super().__init__(train_config, context)

    def _run(self, message: StepMessage) -> StepMessage:
        model: Any = self.get_step_result()
        self.validate_step_result(model, Model)
        (X_train, y_train), (X_val, y_val), _ = message.split.train_val_test  # type: ignore
        model.fit(X_train.collect(), y_train.collect())
        self.card.mod = model
        self.card.mod_outputs = model.get_model_outputs()
        self.card.val_metric = model.score(
            X_val.collect(),
            y_val.collect(),
            metric=get_score_class(self.conf.validation_metric.name),
            **self.conf.validation_metric.params,
        )
        return message


class ClassificationEvaluateStep(EvaluateStep[ClassificationEvaluateConfig]):
    def __init__(self, evaluate_config: ClassificationEvaluateConfig, context: Context):
        super().__init__(evaluate_config, context)

    def _run(self, message: StepMessage) -> StepMessage:
        _, _, (X_test, y_test) = message.split.train_val_test  # type: ignore
        model: Model = message.train.mod  # type: ignore
        metrics_eval: List[Metric] = []
        for criteria in self.conf.validation_criteria:
            score: float = model.score(
                X_test.collect(),
                y_test.collect(),
                metric=get_score_class(criteria.metric.name),
                **criteria.metric.params,
            )
            metrics_eval.append(Metric(name=criteria.metric, value=score))
        self.card.metrics_eval = metrics_eval
        return message


class ClassificationRegisterStep(RegisterStep[ClassificationRegisterConfig]):
    def __init__(self, register_config: ClassificationRegisterConfig, context: Context):
        super().__init__(register_config, context)

    def _run(self, message: StepMessage) -> StepMessage:
        registry: Any = self.get_step_result()
        self.validate_step_result(registry, Registry)
        registry.log_model(message)  # type: ignore
        return message
