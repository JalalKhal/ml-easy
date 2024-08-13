from abc import abstractmethod

from sklearn.metrics import (  # type: ignore
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from ml_easy.recipes.steps.ingest.datasets import Dataset


class Score:
    @classmethod
    @abstractmethod
    def score(cls, y_true: Dataset, y_pred: Dataset, **kwargs) -> float:
        pass


class AccuracyScore(Score):
    @classmethod
    def score(cls, y_true: Dataset, y_pred: Dataset, **kwargs) -> float:
        y_true_np = y_true.to_numpy().flatten()
        y_pred_np = y_pred.to_numpy().flatten()
        return accuracy_score(y_true_np, y_pred_np, **kwargs)


class F1Score(Score):
    @classmethod
    def score(cls, y_true: Dataset, y_pred: Dataset, **kwargs) -> float:
        y_true_np = y_true.to_numpy().flatten()
        y_pred_np = y_pred.to_numpy().flatten()
        return f1_score(y_true_np, y_pred_np, **kwargs)


class AUCScore(Score):
    @classmethod
    def score(cls, y_true: Dataset, y_pred: Dataset, **kwargs) -> float:
        y_true_np = y_true.to_numpy().flatten()
        y_pred_np = y_pred.to_numpy().flatten()
        return roc_auc_score(y_true_np, y_pred_np, **kwargs)


class MAEScore(Score):
    @classmethod
    def score(cls, y_true: Dataset, y_pred: Dataset, **kwargs) -> float:
        y_true_np = y_true.to_numpy().flatten()
        y_pred_np = y_pred.to_numpy().flatten()
        return mean_absolute_error(y_true_np, y_pred_np)


class MSEScore(Score):
    @classmethod
    def score(cls, y_true: Dataset, y_pred: Dataset, **kwargs) -> float:
        y_true_np = y_true.to_numpy().flatten()
        y_pred_np = y_pred.to_numpy().flatten()
        return mean_squared_error(y_true_np, y_pred_np, **kwargs)


class R2Score(Score):
    @classmethod
    def score(cls, y_true: Dataset, y_pred: Dataset, **kwargs) -> float:
        y_true_np = y_true.to_numpy().flatten()
        y_pred_np = y_pred.to_numpy().flatten()
        return r2_score(y_true_np, y_pred_np, **kwargs)
