import importlib
import logging
from typing import Type, TypeVar, Generic

from recipes.interfaces.config import Context
from recipes.interfaces.step import BaseStep
from recipes.steps.cards_config import IngestCard, StepMessage
from recipes.steps.ingest.datasets import Dataset
from recipes.utils import get_step_output_path, get_step_component_output_path

_logger = logging.getLogger(__name__)


U = TypeVar("U", bound="BaseIngestConfig")

class IngestStep(BaseStep[U, IngestCard], Generic[U]):

    def __init__(self, ingest_config: U, context: Context):
        super().__init__(ingest_config, context)

    @property
    def name(self) -> str:
        """
        Returns back the name of the step for the current class instance. This is used
        downstream by the execution engine to create step-specific directory structures.
        """
        return 'ingest'

    def _create_card(self) -> IngestCard:
        step_output_path = get_step_output_path(self.context.recipe_root_path, self.name)
        return IngestCard(step_output_path = step_output_path)




