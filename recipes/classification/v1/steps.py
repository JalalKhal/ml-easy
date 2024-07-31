import os
from typing import Any, List

from recipes.classification.v1.config import ClassificationIngestConfig, ClassificationTransformConfig, \
    ClassificationSplitConfig, ClassificationTrainConfig, ClassificationEvaluateConfig, ClassificationRegisterConfig
from recipes.interfaces.config import Context
from recipes.steps.cards_config import StepMessage, Metric
from recipes.steps.evaluate.evaluate import EvaluateStep
from recipes.steps.ingest.datasets import Dataset
from recipes.steps.ingest.ingest import IngestStep
from recipes.steps.register.ModelRegistry import ModelRegistry
from recipes.steps.register.register_ import RegisterStep
from recipes.steps.split.split import SplitStep
from recipes.steps.split.splitter import DatasetSplitter
from recipes.steps.train.models import Model
from recipes.steps.train.train import TrainStep
from recipes.steps.transform.transform import TransformStep
from recipes.steps.transform.transformer import Transformer
from recipes.utils import get_score_class, get_features_target


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
        transformer: Any = self.get_step_result()
        self.validate_step_result(transformer, Transformer)
        self.card.tf_dataset = transformer.fit_transform(message.ingest.dataset)  # type: ignore
        return message


class ClassificationSplitStep(SplitStep[ClassificationSplitConfig]):
    def __init__(self, split_config: ClassificationSplitConfig, context: Context):
        super().__init__(split_config, context)

    def _run(self, message: StepMessage) -> StepMessage:
        dataset_splitter: Any = self.get_step_result()
        self.validate_step_result(dataset_splitter, DatasetSplitter)
        self.card.train_val_test = dataset_splitter.split(message.transform.tf_dataset)  # type: ignore
        return message


class ClassificationTrainStep(TrainStep[ClassificationTrainConfig]):
    def __init__(self, train_config: ClassificationTrainConfig, context: Context):
        super().__init__(train_config, context)

    def _run(self, message: StepMessage) -> StepMessage:
        model: Any = self.get_step_result()
        self.validate_step_result(model, Model)
        train: Dataset = message.split.train_val_test[0].collect()  # type: ignore
        val: Dataset = message.split.train_val_test[1].collect()  # type: ignore
        X_train, y_train = get_features_target(train, self.context.target_col)
        X_val, y_val = get_features_target(val, self.context.target_col)
        model.fit(X_train, y_train)
        self.card.mod = model
        self.card.mod_outputs = model.get_model_outputs()
        self.card.val_metric = model.score(X_val, y_val,
                                           metric=get_score_class(self.conf.validation_metric.name),
                                           **self.conf.validation_metric.params)
        return message


class ClassificationEvaluateStep(EvaluateStep[ClassificationEvaluateConfig]):
    def __init__(self, evaluate_config: ClassificationEvaluateConfig, context: Context):
        super().__init__(evaluate_config, context)

    def _run(self, message: StepMessage) -> StepMessage:
        test: Dataset = message.split.train_val_test[0].collect()  # type: ignore
        X_test, y_test = get_features_target(test, self.context.target_col)
        model: Model = message.train.mod  # type: ignore
        metrics_eval: List[Metric] = []
        for criteria in self.conf.validation_criteria:
            score: float = model.score(X_test, y_test,
                                       metric=get_score_class(criteria.metric.name),
                                       **criteria.metric.params)
            metrics_eval.append(Metric(name=criteria.metric, value=score))
        self.card.metrics_eval = metrics_eval
        return message


class ClassificationRegisterStep(RegisterStep[ClassificationRegisterConfig]):
    def __init__(self, register_config: ClassificationRegisterConfig, context: Context):
        super().__init__(register_config, context)

    def _run(self, message: StepMessage) -> StepMessage:
        registry: Any = self.get_step_result()
        self.validate_step_result(registry, ModelRegistry)
        artifact_path: str = f"{self.card.step_output_path}{self.conf.artifact_path}"
        registry.log_model(message.train.mod, artifact_path)  # type: ignore
        self.card.artifact_path = artifact_path
        return message
