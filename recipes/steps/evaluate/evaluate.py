import logging
from typing import TypeVar, Generic, Type, Optional

from recipes.interfaces.config import Context
from recipes.interfaces.step import BaseStep
from recipes.steps.cards_config import EvaluateCard
from recipes.steps.steps_config import BaseEvaluateConfig
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

    @classmethod
    def card_type(cls) -> Type[EvaluateCard]:
        """
        Returns the type of card to be created for the step.
        """
        return EvaluateCard
    @property
    def previous_step_name(self) -> Optional[str]:
        return 'train'




