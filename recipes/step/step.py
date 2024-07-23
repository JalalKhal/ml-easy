import abc
import json
import logging
import os
from datetime import time
from typing import Dict, Any, Optional

from recipes.step.enum import StepStatus, StepExecutionStateKeys
from recipes.utils import get_recipe_name

_logger = logging.getLogger(__name__)



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
            StepExecutionStateKeys.KEY_STATUS: self.status.value,
            StepExecutionStateKeys.KEY_LAST_UPDATED_TIMESTAMP: self.last_updated_timestamp,
            StepExecutionStateKeys.KEY_STACK_TRACE: self.stack_trace,
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


class BaseStep(metaclass=abc.ABCMeta):
    """
    Base class representing a step in an MLflow Recipe
    """

    _EXECUTION_STATE_FILE_NAME = "execution_state.json"

    def __init__(self, step_config: Dict[str, Any], recipe_root: str):
        """
        Args:
            step_config: Dictionary of the config needed to run/implement the step.
            recipe_root: String file path to the directory where step are defined.
        """
        self.step_config = step_config
        self.recipe_root = recipe_root
        self.recipe_name = get_recipe_name(recipe_root_path=recipe_root)
        self.task = self.step_config.get("recipe", "regression/v1").rsplit("/", 1)[0]
        self.step_card = None

    def __str__(self):
        return f"Step:{self.name}"

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Returns back the name of the step for the current class instance. This is used
        downstream by the execution engine to create step-specific directory structures.
        """

    def run(self, output_directory: str):
        """
        Executes the step by running common setup operations and invoking
        step-specific code (as defined in ``_run()``).

        Args:
            output_directory: String file path to the directory where step
                outputs should be stored.
        """
        _logger.info(f"Running step {self.name}...")
        start_timestamp = time.time()
        try:
            self._update_status(status=StepStatus.RUNNING, output_directory=output_directory)



            self._validate_and_apply_step_config()
            self.step_card = self._run(output_directory=output_directory)
            self._update_status(status=StepStatus.SUCCEEDED, output_directory=output_directory)
        except Exception:
            stack_trace = traceback.format_exc()
            self._update_status(
                status=StepStatus.FAILED, output_directory=output_directory, stack_trace=stack_trace
            )
            self.step_card = FailureCard(
                recipe_name=self.recipe_name,
                step_name=self.name,
                failure_traceback=stack_trace,
                output_directory=output_directory,
            )
            raise
        finally:
            self._serialize_card(start_timestamp, output_directory)

    def _update_status(
            self, status: StepStatus, output_directory: str, stack_trace: Optional[str] = None
    ) -> None:
        execution_state = StepExecutionState(
            status=status, last_updated_timestamp=time.time(), stack_trace=stack_trace
        )
        with open(os.path.join(output_directory, BaseStep._EXECUTION_STATE_FILE_NAME), "w") as f:
            json.dump(execution_state.to_dict(), f)


