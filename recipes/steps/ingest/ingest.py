import abc
import logging
from typing import Any

from recipes.interfaces.step import BaseStep
from recipes.steps.cards_config import IngestCard, Context
from recipes.steps.steps_config import IngestConfig
from recipes.utils import get_step_output_path

_logger = logging.getLogger(__name__)




class IngestStep(BaseStep[IngestConfig, IngestCard], metaclass=abc.ABCMeta):
    def __init__(self, ingest_config: IngestConfig, context: Context):
        super().__init__(ingest_config)
        self._context = context

    @property
    def name(self) -> str:
        """
        Returns back the name of the step for the current class instance. This is used
        downstream by the execution engine to create step-specific directory structures.
        """
        return 'ingest'

    def _create_card(self) -> IngestCard:
        step_output_path = get_step_output_path(self._context.recipe_root_path, self.name)
        return IngestCard(step_output_path = step_output_path)

    def _run(self, card: Any) -> IngestCard:
        return self.card








