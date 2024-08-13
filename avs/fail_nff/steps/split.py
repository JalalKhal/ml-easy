"""
This module defines the following routines used by the 'split' step of the regression recipe:

- ``create_dataset_filter``: Defines customizable logic for filtering the training
  datasets produced by the data splitting procedure. Note that arbitrary transformations
  should go into the transform step.
"""

from ml_easy.recipes.classification.v1.config import ClassificationSplitConfig
from ml_easy.recipes.interfaces.config import Context
from ml_easy.recipes.steps.split.splitter import DatasetSplitter


def split_fn(conf: ClassificationSplitConfig, context: Context) -> DatasetSplitter:
    return DatasetSplitter(conf.split_ratios[1], conf.split_ratios[2])
