import json
import pickle
from abc import abstractmethod

import mlflow
import numpy as np
import requests
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion

from recipes.steps.ingest.datasets import Dataset, PolarsDataset
from recipes.steps.transform.transformer import Transformer


class Inference:
    @abstractmethod
    def predict(self, X: Dataset) -> Dataset:
        pass


class MlflowInference(Inference):

    def __init__(self, tracking_uri: str, serving_port: int, registered_model_name: str):
        mlflow.set_tracking_uri(tracking_uri)
        self.tracking_client = MlflowClient()
        self.registered_model_name = registered_model_name
        self.transformer: Transformer = self.load_transformer()
        self.serving_uri = f"{':'.join(tracking_uri.split(':')[:2])}:{str(serving_port)}/invocations"

    def get_model_info(self) -> ModelVersion:
        model_info = self.tracking_client.get_model_version(*(self.registered_model_name).split(':'))
        return model_info

    def load_transformer(self) -> Transformer:
        run_id = self.get_model_info().run_id
        transformer_path = self.tracking_client.download_artifacts(run_id, 'transformer')
        file = open(f"{transformer_path}/transformer.pkl", 'rb')
        return pickle.load(file)

    def predict(self, X: Dataset) -> Dataset:
        tf_X: Dataset = self.transformer.transform(X)
        response = requests.post(
            url=self.serving_uri,
            data=json.dumps({'instances': tf_X.to_numpy().tolist()}),
            headers={'Content-Type': 'application/json'},
        )
        predictions: Dataset = PolarsDataset.from_numpy(np.array(response.json()['predictions']))
        return predictions


if __name__ == '__main__':
    inf = MlflowInference('http://127.0.0.1:5000', 5001, 'registered_model_name:11')
    predictions = inf.predict(
        PolarsDataset.read_csv(
            '/home/khaldi/Documents/github_repos/refined_mlflow/avs/fail_nff/sru_datatset.csv',
            separator=',',
            encoding='ISO-8859-1',
        ).slice(0, 5)
    )
    print(predictions)
