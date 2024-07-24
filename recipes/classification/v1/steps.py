from typing import Self

from recipes.classification.v1.config import ClassificationIngestConfig
from recipes.interfaces.config import BaseStepConfig, Context
from recipes.steps.ingest.ingest import IngestStep


class ClassificationIngestStep(IngestStep[ClassificationIngestConfig]):
    def __init__(self, ingest_config: ClassificationIngestConfig, context: Context):
        super().__init__(ingest_config, context)

    @classmethod
    def from_recipe_config(cls, config: BaseStepConfig, context: Context) -> Self:
        if not isinstance(config, ClassificationIngestConfig):
            raise TypeError(f"{config.__class__.__name__} should be an instance of {ClassificationIngestConfig}")
        return cls(config, context)
