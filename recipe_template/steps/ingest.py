"""
This module defines the following routines used by the 'ingest' step of the regression recipe:

- ``load_file_as_dataframe``: Defines customizable logic for parsing dataset formats that are not
  natively parsed by MLflow Recipes (i.e. formats other than Parquet, Delta, and Spark SQL).
"""
from pandas import DataFrame

from recipes.classification.v1.config import ClassificationIngestConfig
from recipes.interfaces.config import Context
from recipes.steps.ingest.datasets import PolarsDataset, Dataset


def ingest_fn(conf: ClassificationIngestConfig, context: Context) -> Dataset:
    return PolarsDataset.read_csv(
        conf.location,
        conf.sep,
        conf.encoding
    )



