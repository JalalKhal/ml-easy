"""
This module defines the following routines used by the 'transform' step of the regression recipe:

- ``transformer_fn``: Defines customizable logic for transforming input data before it is passed
  to the estimator during model inference.
"""

from recipes.classification.v1.config import ClassificationTransformConfig
from recipes.constants import FILTER_TO_MODULE
from recipes.interfaces.config import Context
from recipes.steps.transform.filters import Filter
from recipes.steps.transform.transformer import (
    FilterTransformer,
    MultipleTfIdfTransformer,
    PipelineTransformer,
    Transformer,
)


def transformer_fn(conf: ClassificationTransformConfig, context: Context) -> Transformer:
    """
    Returns an *unfitted* transformer that defines ``fit()`` and ``transform()`` methods.
    The transformer's input and output signatures should be compatible with scikit-learn
    transformers.
    """

    return PipelineTransformer(
        [
            FilterTransformer(
                {
                    col: [
                        Filter.load_from_path(FILTER_TO_MODULE[filter_conf.type])(
                            **filter_conf.model_dump(exclude={'type'})
                        )
                        for filter_conf in conf.cols[col].filters
                    ]
                    for col in conf.cols
                    if conf.cols[col].filters
                }
            ),
            MultipleTfIdfTransformer(conf, context),
        ]
    )
