import abc
import logging
from typing import TypeVar, Generic, Type, Optional

from recipes.interfaces.config import Context
from recipes.interfaces.step import BaseStep
from recipes.steps.cards_config import StepMessage, TransformCard, TrainCard
from recipes.steps.steps_config import BaseTrainConfig
from recipes.utils import get_step_output_path

_logger = logging.getLogger(__name__)


U = TypeVar("U", bound="BaseTrainConfig")


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



