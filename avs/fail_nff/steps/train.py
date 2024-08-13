"""
This module defines the following routines used by the 'train' step of the regression recipe:

- ``estimator_fn``: Defines the customizable estimator type and parameters that are used
  during training to produce a model recipe.
"""

from ml_easy.recipes.classification.v1.config import ClassificationTrainConfig
from ml_easy.recipes.interfaces.config import Context
from ml_easy.recipes.steps.train.models import Model, ScikitModel


def estimator_fn(conf: ClassificationTrainConfig, context: Context) -> Model:
    """
    Returns an *unfitted* estimator that defines ``fit()`` and ``predict()`` methods.
    The estimator's input and output signatures should be compatible with scikit-learn
    estimators.
    """
    path = 'sklearn.linear_model.LogisticRegression'
    estimator: ScikitModel = ScikitModel.load_from_library(path, {'max_iter': 3000})
    return estimator
