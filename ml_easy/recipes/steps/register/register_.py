import logging
from typing import Generic, Optional, Type, TypeVar

from ml_easy.recipes.interfaces.config import Context
from ml_easy.recipes.interfaces.step import BaseStep
from ml_easy.recipes.steps.cards_config import RegisterCard
from ml_easy.recipes.steps.steps_config import BaseRegisterConfig

_logger = logging.getLogger(__name__)

U = TypeVar('U', bound='BaseRegisterConfig')


class RegisterStep(BaseStep[U, RegisterCard], Generic[U]):

    def __init__(self, register_config: U, context: Context):
        super().__init__(register_config, context)

    @property
    def name(self) -> str:
        """
        Returns back the name of the step for the current class instance. This is used
        downstream by the execution engine to create step-specific directory structures.
        """
        return 'register_'

    @classmethod
    def card_type(cls) -> Type[RegisterCard]:
        """
        Returns the type of card to be created for the step.
        """
        return RegisterCard

    @property
    def previous_step_name(self) -> Optional[str]:
        return 'evaluate'
