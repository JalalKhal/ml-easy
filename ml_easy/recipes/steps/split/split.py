import logging
from abc import ABC
from typing import Generic, Optional, Type, TypeVar

from ml_easy.recipes.interfaces.config import Context
from ml_easy.recipes.interfaces.step import BaseStep
from ml_easy.recipes.steps.cards_config import SplitCard
from ml_easy.recipes.steps.steps_config import BaseSplitConfig

_logger = logging.getLogger(__name__)

U = TypeVar('U', bound='BaseSplitConfig')


class SplitStep(BaseStep[U, SplitCard], Generic[U], ABC):

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

    @property
    def previous_step_name(self) -> Optional[str]:
        return 'transform'
