import abc
import logging
from typing import TypeVar, Generic

from recipes.interfaces.config import Context
from recipes.interfaces.step import BaseStep
from recipes.steps.cards_config import StepMessage, TransformCard
from recipes.utils import get_step_output_path

_logger = logging.getLogger(__name__)


U = TypeVar("U", bound="BaseTransformConfig")

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

    def _create_card(self) -> TransformCard:
        step_output_path = get_step_output_path(self.context.recipe_root_path, self.name)
        return TransformCard(step_output_path = step_output_path)



