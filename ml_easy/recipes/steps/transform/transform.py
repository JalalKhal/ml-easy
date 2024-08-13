import logging
from typing import Generic, Optional, Type, TypeVar

from ml_easy.recipes.interfaces.config import Context
from ml_easy.recipes.interfaces.step import BaseStep
from ml_easy.recipes.steps.cards_config import TransformCard
from ml_easy.recipes.steps.steps_config import BaseTransformConfig

_logger = logging.getLogger(__name__)


U = TypeVar('U', bound='BaseTransformConfig')


class TransformStep(BaseStep[U, TransformCard], Generic[U]):

    def __init__(self, transform_config: U, context: Context):
        super().__init__(transform_config, context)

    @property
    def name(self) -> str:
        """
        Returns back the name of the step for the current class instance. This is used
        downstream by the execution engine to create step-specific directory structures.
        """
        return 'transform'

    @classmethod
    def card_type(cls) -> Type[TransformCard]:
        """
        Returns the type of card to be created for the step.
        """
        return TransformCard

    @property
    def previous_step_name(self) -> Optional[str]:
        return 'ingest'
