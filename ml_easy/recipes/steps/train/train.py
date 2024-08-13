import logging
from typing import Generic, Optional, Type, TypeVar

from ml_easy.recipes.interfaces.config import Context
from ml_easy.recipes.interfaces.step import BaseStep
from ml_easy.recipes.steps.cards_config import TrainCard
from ml_easy.recipes.steps.steps_config import BaseTrainConfig

_logger = logging.getLogger(__name__)


U = TypeVar('U', bound='BaseTrainConfig')


class TrainStep(BaseStep[U, TrainCard], Generic[U]):

    def __init__(self, train_config: U, context: Context):
        super().__init__(train_config, context)

    @property
    def name(self) -> str:
        """
        Returns back the name of the step for the current class instance. This is used
        downstream by the execution engine to create step-specific directory structures.
        """
        return 'train'

    @classmethod
    def card_type(cls) -> Type[TrainCard]:
        """
        Returns the type of card to be created for the step.
        """
        return TrainCard

    @property
    def previous_step_name(self) -> Optional[str]:
        return 'split'
