import logging
from typing import TypeVar, Generic

from recipes.interfaces.config import Context
from recipes.interfaces.step import BaseStep
from recipes.steps.cards_config import TransformCard, SplitCard
from recipes.utils import get_step_output_path

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

    def _create_card(self) -> SplitCard:
        step_output_path = get_step_output_path(self.context.recipe_root_path, self.name)
        return SplitCard(step_output_path=step_output_path)
