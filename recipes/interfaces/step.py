import abc
import json
import logging
import os
import time
import traceback
from typing import Dict, Any, Optional, TypeVar, Generic

from recipes.enum import StepExecutionStateKeys, StepStatus
from recipes.interfaces.config import Context
from recipes.steps.cards_config import StepMessage

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

    def run(self, message: StepMessage) -> StepMessage:

        _logger.info(f"Running step {self.name}...")
        try:
            self._update_status(status=StepStatus.RUNNING, output_directory=self.card.step_output_path)
            message = self._run(message)
            self._update_status(status=StepStatus.SUCCEEDED, output_directory=self.card.step_output_path)
            return message
        except Exception:
            stack_trace = traceback.format_exc()
            self._update_status(
                status=StepStatus.FAILED, output_directory=self.card.step_output_path, stack_trace=stack_trace
            )
            raise

    def _update_status(
            self, status: StepStatus, output_directory: str, stack_trace: Optional[str] = None
    ) -> None:
        execution_state = StepExecutionState(
            status=status, last_updated_timestamp=time.time(), stack_trace=stack_trace
        )
        with open(os.path.join(output_directory, BaseStep._EXECUTION_STATE_FILE_NAME), "w") as f:
            json.dump(execution_state.to_dict(), f)

    @abc.abstractmethod
    def _run(self, message: StepMessage) -> StepMessage:
        """
        This function is responsible for executing the step, writing outputs
        to the specified directory, and returning results to the user. It
        is invoked by the internal step runner.

        Args:
            output_directory: String file path to the directory where step outputs
                should be stored.
        """

    @abc.abstractmethod
    def _create_card(self) -> V:
        pass





