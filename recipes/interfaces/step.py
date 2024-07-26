import abc
import json
import logging
import os
import time
import traceback
from typing import Dict, Any, Optional, TypeVar, Generic, Type

from recipes.enum import StepExecutionStateKeys, StepStatus, MLFlowErrorCode
from recipes.exceptions import MlflowException
from recipes.interfaces.config import Context
from recipes.steps.cards_config import StepMessage
from recipes.utils import get_fully_qualified_module_name_for_step, load_step_function, get_step_fn, \
    get_step_output_path

_logger = logging.getLogger(__name__)

U = TypeVar('U', bound="BaseStepConfig")
V = TypeVar('V', bound="BaseCard")


class StepExecutionState:
    """
    Represents execution state for a step, including the current status and
    the time of the last status update.
    """

    def __init__(self, status: StepStatus, last_updated_timestamp: int, stack_trace: str):
        """
        Args:
            status: The execution status of the step.
            last_updated_timestamp: The timestamp of the last execution status update, measured
                in seconds since the UNIX epoch.
            stack_trace: The stack trace of the last execution. None if the step execution
                succeeds.
        """
        self.status = status
        self.last_updated_timestamp = last_updated_timestamp
        self.stack_trace = stack_trace

    def to_dict(self) -> Dict[StepExecutionStateKeys, Any]:
        """
        Creates a dictionary representation of the step execution state.
        """
        return {
            StepExecutionStateKeys.KEY_STATUS.name: self.status.value,
            StepExecutionStateKeys.KEY_LAST_UPDATED_TIMESTAMP.name: self.last_updated_timestamp,
            StepExecutionStateKeys.KEY_STACK_TRACE.name: self.stack_trace,
        }

    @classmethod
    def from_dict(cls, state_dict) -> "StepExecutionState":
        """
        Creates a ``StepExecutionState`` instance from the specified execution state dictionary.
        """
        return cls(
            status=StepStatus[state_dict[StepExecutionStateKeys.KEY_STATUS]],
            last_updated_timestamp=state_dict[StepExecutionStateKeys.KEY_LAST_UPDATED_TIMESTAMP],
            stack_trace=state_dict[StepExecutionStateKeys.KEY_STACK_TRACE],
        )


class BaseStep(Generic[U, V], metaclass=abc.ABCMeta):
    """
    Base class representing a step in an MLflow Recipe
    """

    _EXECUTION_STATE_FILE_NAME = "execution_state.json"
    _CUSTOM_STEPS_DIR = "steps"
    _SUFFIX_FN = "_fn"

    def __init__(self, step_config: U, context: Context):
        """
        Args:
            step_config: Dictionary of the config needed to run/implement the step.
            recipe_root: String file path to the directory where step are defined.
        """
        self.conf = step_config
        self.context = context
        self.card: V = self._create_card()

    def __str__(self):
        return f"Step:{self.name}"

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Returns back the name of the step for the current class instance. This is used
        downstream by the execution engine to create step-specific directory structures.
        """

    @classmethod
    @abc.abstractmethod
    def card_type(cls) -> Type[V]:
        """
        Returns the type of card to be created for the step.
        """

    def run(self, message: StepMessage) -> StepMessage:

        _logger.info(f"Running step {self.name}...")
        try:
            self._update_status(status=StepStatus.RUNNING, output_directory=self.card.step_output_path)
            message = self._run(message)
            self.update_message(message)
            self._update_status(status=StepStatus.SUCCEEDED, output_directory=self.card.step_output_path)
            return message
        except Exception:
            stack_trace = traceback.format_exc()
            self._update_status(
                status=StepStatus.FAILED, output_directory=self.card.step_output_path, stack_trace=stack_trace
            )
            raise

    @abc.abstractmethod
    def _run(self, message: StepMessage) -> StepMessage:
        pass

    def _update_status(
            self, status: StepStatus, output_directory: str, stack_trace: Optional[str] = None
    ) -> None:
        execution_state = StepExecutionState(
            status=status, last_updated_timestamp=time.time(), stack_trace=stack_trace
        )
        with open(os.path.join(output_directory, BaseStep._EXECUTION_STATE_FILE_NAME), "w") as f:
            json.dump(execution_state.to_dict(), f)

    def get_step_result(self, from_fn=True) -> Any:
        if from_fn:
            step_fn = get_step_fn(self.conf, self._SUFFIX_FN)
            step_result: Any = load_step_function(self.get_module_name_for_step_function(), step_fn)(self.conf,
                                                                                                     self.context)
            return step_result

    @classmethod
    def validate_step_result(cls, obj: Any, class_: Type[Any]):
        if not isinstance(obj, class_):
            raise MlflowException(
                f"{obj.__class__.__name__} should be a {class_} instance",
                error_code=MLFlowErrorCode.INTERNAL_ERROR,
            ) from None

    def _create_card(self) -> V:
        step_output_path = get_step_output_path(self.context.recipe_root_path, self.name)
        return self.card_type()(step_output_path=step_output_path)

    def get_module_name_for_step_function(self) -> str:
        return get_fully_qualified_module_name_for_step(
            self.context.recipe_root_path,
            self._CUSTOM_STEPS_DIR,
            self.name)

    def update_message(self, message: StepMessage) -> None:
        setattr(message, self.name, self.card)


