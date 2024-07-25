"""
This module defines the following routines used by the 'transform' step of the regression recipe:

- ``transformer_fn``: Defines customizable logic for transforming input data before it is passed
  to the estimator during model inference.
"""
from typing import Type

from recipes.classification.v1.config import ClassificationTransformConfig
from recipes.interfaces.config import Context
from recipes.steps.ingest.datasets import Dataset
from recipes.steps.transform.transformer import Transformer


def transformer_fn(conf: ClassificationTransformConfig, context: Context) -> Transformer:
    """
    Returns an *unfitted* transformer that defines ``fit()`` and ``transform()`` methods.
    The transformer's input and output signatures should be compatible with scikit-learn
    transformers.
    """
    #
    # FIXME::OPTIONAL: return a scikit-learn-compatible transformer object.
    #
    # Identity feature transformation is applied when None is returned.
    class TemplateTransformer(Transformer):
        def fit(self, X: Dataset) -> None:
            pass
        def transform(self, X: Dataset) -> Dataset:
            return X

    return TemplateTransformer()

