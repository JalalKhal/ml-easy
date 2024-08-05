from abc import abstractmethod

import mlflow
from mlflow.models import infer_signature

from recipes.interfaces.config import Context
from recipes.steps.cards_config import StepMessage
from recipes.steps.steps_config import BaseRegisterConfig
from recipes.steps.train.models import ScikitModel
from recipes.utils import get_features_target


class Registry:
    def __init__(self):
        pass

    @abstractmethod
    def log_model(self, message: StepMessage) -> None:
        pass

    @abstractmethod
    def log_embedder(self, message: StepMessage) -> None:
        pass


class MlflowRegistry(Registry):

    def __init__(self, conf: BaseRegisterConfig, context: Context):
        super().__init__()
        self.context = context
        self.conf = conf

    def log_embedder(self, message: StepMessage) -> None:
        mlflow.log_artifact(message.transform.transformer_path, 'transformer')

    def log_model(self, message: StepMessage) -> None:
        mlflow.set_tracking_uri(self.context.experiment.tracking_uri)
        mlflow.set_experiment(self.context.experiment.name)
        with mlflow.start_run():
            self.log_embedder(message)
            if isinstance(message.train.mod, ScikitModel):
                train, validation, test = message.split.train_val_test
                X_test, y_test = get_features_target(test, self.context.target_col)
                signature = infer_signature(X_test.to_numpy(), message.train.mod.predict(X_test).to_numpy().reshape(-1))
                mlflow.sklearn.log_model(
                    message.train.mod._service,
                    self.conf.artifact_path,
                    signature=signature,
                    registered_model_name=self.conf.registered_model_name,
                )
                mlflow.log_params(dict(message.transform.config))
                mlflow.log_metrics({m.name.name.value: m.value for m in message.evaluate.metrics_eval})
                mlflow.set_tag('product_name', self.context.experiment.product_name)
        mlflow.end_run()
