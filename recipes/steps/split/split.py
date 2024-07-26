import logging
from typing import TypeVar, Generic, Type

from recipes.interfaces.config import Context
from recipes.interfaces.step import BaseStep
from recipes.steps.cards_config import SplitCard

_logger = logging.getLogger(__name__)

U = TypeVar("U", bound="BaseSplitConfig")


class SplitStep(BaseStep[U, SplitCard], Generic[U]):

    def __init__(self, split_config: U, context: Context):
        super().__init__(split_config, context)

    @property
    def name(self) -> str:
        """
        Returns back the name of the step for the current class instance. This is used
        downstream by the execution engine to create step-specific directory structures.
        """
        return 'split'

    @classmethod
    def card_type(cls) -> Type[SplitCard]:
        """
        Returns the type of card to be created for the step.
        """
        return SplitCard
