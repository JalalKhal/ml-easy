from abc import abstractmethod

import mlflow  # type:ignore
from mlflow.data.code_dataset_source import CodeDatasetSource  # type: ignore
from mlflow.models import infer_signature  # type:ignore

from ml_easy.recipes.interfaces.config import Context
from ml_easy.recipes.steps.cards_config import StepMessage
from ml_easy.recipes.steps.steps_config import BaseRegisterConfig
from ml_easy.recipes.steps.train.models import ScikitModel


class Registry:
    def __init__(self):
        pass

    @abstractmethod
    def log_model(self, message: StepMessage) -> None:
        pass

    @abstractmethod
    def log_embedder(self, message: StepMessage) -> None:
        pass

    @abstractmethod
    def log_dataset(self, message: StepMessage) -> None:
        pass


class MlflowRegistry(Registry):

    def __init__(self, conf: BaseRegisterConfig, context: Context):
        super().__init__()
        self.context = context
        self.conf = conf

    def log_embedder(self, message: StepMessage) -> None:
        mlflow.log_artifact(message.transform.transformer_path, 'transformer')  # type:ignore

    def log_dataset(self, message: StepMessage) -> None:
        (X, y) = message.transform.tf_dataset  # type: ignore
        mlflow.log_input(X.get_mlflow_dataset(self.conf.source))  # type: ignore
        mlflow.log_input(y.get_mlflow_dataset(self.conf.source))  # type: ignore

    def log_model(self, message: StepMessage) -> None:
        mlflow.set_tracking_uri(self.context.experiment.tracking_uri)  # type:ignore
        mlflow.set_experiment(self.context.experiment.name)  # type:ignore
        with mlflow.start_run():
            self.log_embedder(message)
            self.log_dataset(message)
            if isinstance(message.train.mod, ScikitModel):  # type:ignore
                _, _, (X_test, y_test) = message.split.train_val_test  # type: ignore
                signature = infer_signature(
                    X_test[:3, :].to_numpy(), message.train.mod.predict(X_test[:3, :]).to_numpy().reshape(-1)  # type: ignore
                )  # type:ignore
                mlflow.sklearn.log_model(
                    message.train.mod.service,  # type:ignore
                    self.conf.artifact_path,  # type:ignore
                    signature=signature,
                    registered_model_name=self.conf.registered_model_name,  # type:ignore
                )
                mlflow.log_params(dict(message.transform.config))  # type:ignore
                mlflow.log_metrics({m.name.name.value: m.value for m in message.evaluate.metrics_eval})  # type:ignore
                mlflow.set_tag('product_name', self.context.experiment.product_name)
        mlflow.end_run()
