import logging
from typing import Generic, Optional, Type, TypeVar

from ml_easy.recipes.interfaces.config import Context
from ml_easy.recipes.interfaces.step import BaseStep
from ml_easy.recipes.steps.cards_config import IngestCard
from ml_easy.recipes.steps.steps_config import BaseIngestConfig

_logger = logging.getLogger(__name__)


U = TypeVar('U', bound='BaseIngestConfig')


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

    @classmethod
    def card_type(cls) -> Type[IngestCard]:
        """
        Returns the type of card to be created for the step.
        """
        return IngestCard

    @property
    def previous_step_name(self) -> Optional[str]:
        return None
