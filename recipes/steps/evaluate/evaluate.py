import logging
from typing import TypeVar, Generic

from recipes.interfaces.config import Context
from recipes.interfaces.step import BaseStep
from recipes.steps.cards_config import EvaluateCard
from recipes.utils import get_step_output_path

_logger = logging.getLogger(__name__)


U = TypeVar("U", bound="BaseEvaluateConfig")

class EvaluateStep(BaseStep[U, EvaluateCard], Generic[U]):

    def __init__(self, evaluate_config: U, context: Context):
        super().__init__(evaluate_config, context)

    @property
    def name(self) -> str:
        """
        Returns back the name of the step for the current class instance. This is used
        downstream by the execution engine to create step-specific directory structures.
        """
        return 'evaluate'

    def _create_card(self) -> EvaluateCard:
        step_output_path = get_step_output_path(self.context.recipe_root_path, self.name)
        return EvaluateCard(step_output_path = step_output_path)




