"""
This module defines the following routines used by the 'transform' step of the regression recipe:

- ``transformer_fn``: Defines customizable logic for transforming input data before it is passed
  to the estimator during model inference.
"""

from recipes.classification.v1.config import ClassificationTransformConfig
from recipes.interfaces.config import Context
from recipes.steps.transform.transformer import MultipleTfIdfTransformer, Transformer


def transformer_fn(conf: ClassificationTransformConfig, context: Context) -> Transformer:
    """
    Returns an *unfitted* transformer that defines ``fit()`` and ``transform()`` methods.
    The transformer's input and output signatures should be compatible with scikit-learn
    transformers.
    """

    return MultipleTfIdfTransformer(conf, context)
