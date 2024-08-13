import hashlib
import importlib
import os
from typing import Any, Tuple, Type

from typeguard import TypeCheckError, check_type

from ml_easy.recipes.constants import (
    EXT_PY,
    SCORES_PATH,
    SOURCE_TO_MODULE,
    STEP_OUTPUTS_SUBDIRECTORY_NAME,
    STEPS_SUBDIRECTORY_NAME,
)
from ml_easy.recipes.enum import MLFlowErrorCode, ScoreType
from ml_easy.recipes.env_vars import MLFLOW_RECIPES_EXECUTION_DIRECTORY
from ml_easy.recipes.exceptions import MlflowException
from ml_easy.recipes.steps.evaluate.score import Score
from ml_easy.recipes.steps.ingest.datasets import Dataset
from ml_easy.recipes.steps.register.mlflow_source.sql_table_dataset_source import (
    DatasetSourceWrapper,
)
from ml_easy.recipes.steps.steps_config import SourceConfig


def get_recipe_name(recipe_root_path: str) -> str:
    """
    Obtains the name of the specified recipe or of the recipe corresponding to the current
    working directory.

    Args:
        recipe_root_path: The absolute path of the recipe root directory on the local
            filesystem. If unspecified, the recipe root directory is resolved from the current
            working directory.

    Raises:
        MlflowException: If the specified ``recipe_root_path`` is not a recipe root
            directory or if ``recipe_root_path`` is ``None`` and the current working directory
            does not correspond to a recipe.

    Returns:
        The name of the specified recipe.
    """
    return os.path.basename(recipe_root_path)


def get_class_from_string(fully_qualified_class_name) -> Any:
    module, class_name = fully_qualified_class_name.rsplit('.', maxsplit=1)
    return getattr(importlib.import_module(module), class_name)


def load_class(fully_qualified_class_name: str) -> Any:
    try:
        class_module = get_class_from_string(fully_qualified_class_name)
    except Exception as e:
        if isinstance(e, ModuleNotFoundError):
            raise MlflowException(
                f"Failed to find {fully_qualified_class_name}.",
                error_code=MLFlowErrorCode.INVALID_PARAMETER_VALUE,
            ) from None
        else:
            raise MlflowException(
                f"Failed to construct {fully_qualified_class_name}. Error: {e!r}",
                error_code=MLFlowErrorCode.INVALID_PARAMETER_VALUE,
            ) from None
    return class_module


def get_fully_qualified_module_name_for_step(recipe_root_path: str, step_dir: str, step_file_name: str):
    fully_qualified_item: str = os.path.join(recipe_root_path, step_dir, step_file_name + EXT_PY)
    return fully_qualified_item


def load_step_function(file_path: str, function_name: str) -> Any:
    if not os.path.exists(file_path):
        raise MlflowException(
            f"File {file_path} not found.",
            error_code=MLFlowErrorCode.INVALID_PARAMETER_VALUE,
        )

    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise MlflowException(
            f"Could not load function {function_name} into a module", error_code=MLFlowErrorCode.INTERNAL_ERROR
        )
    else:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    try:
        return getattr(module, function_name)
    except AttributeError:
        raise MlflowException(
            f"Function {function_name} not found in file {file_path}.",
            error_code=MLFlowErrorCode.INVALID_PARAMETER_VALUE,
        )


def load_config(obj: Any, config: Any) -> None:
    for field, value in config.__dict__.items():
        setattr(obj, field, value)


def _get_execution_directory_basename(recipe_root_path) -> str:
    """
    Obtains the basename of the execution directory corresponding to the specified recipe.

    Args:
        recipe_root_path: The absolute path of the recipe root directory on the local
            filesystem.

    Returns:
        The basename of the execution directory corresponding to the specified recipe.
    """
    return hashlib.sha256(os.path.abspath(recipe_root_path).encode('utf-8')).hexdigest()


def get_or_create_base_execution_directory(recipe_root_path: str) -> str:
    """
    Obtains the path of the execution directory on the local filesystem corresponding to the
    specified recipe. The directory is created if it does not exist.

    Args:
        recipe_root_path: The absolute path of the recipe root directory on the local
            filesystem.

    Returns:
        The path of the execution directory on the local filesystem corresponding to the
        specified recipe.
    """
    execution_directory_basename = _get_execution_directory_basename(recipe_root_path=recipe_root_path)

    execution_dir_path = os.path.abspath(
        MLFLOW_RECIPES_EXECUTION_DIRECTORY.get()
        or os.path.join(os.path.expanduser('~'), '.mlflow', 'recipes', execution_directory_basename)
    )
    os.makedirs(execution_dir_path, exist_ok=True)
    return execution_dir_path


def _get_step_output_directory_path(execution_directory_path: str, step_name: str) -> str:
    """
    Obtains the path of the local filesystem directory containing outputs for the specified step,
    which may or may not exist.

    Args:
        execution_directory_path: The absolute path of the execution directory on the local
            filesystem for the relevant recipe. The Makefile is created in this directory.
        step_name: The name of the recipe step for which to obtain the output directory path.

    Returns:
        The absolute path of the local filesystem directory containing outputs for the specified
        step.
    """
    return os.path.abspath(
        os.path.join(
            execution_directory_path,
            STEPS_SUBDIRECTORY_NAME,
            step_name,
            STEP_OUTPUTS_SUBDIRECTORY_NAME,
        )
    )


def is_instance_for_generic(obj: Any, _class: Any) -> bool:
    try:
        check_type(obj, _class)
        return True
    except TypeCheckError:
        return False


def get_step_output_path(recipe_root_path: str, step_name: str, relative_path: str = '') -> str:
    """
    Obtains the absolute path of the specified step output on the local filesystem. Does
    not check the existence of the output.

    Args:
        recipe_root_path: The absolute path of the recipe root directory on the local
            filesystem.
        step_name: The name of the recipe step containing the specified output.
        relative_path: The relative path of the output within the output directory
            of the specified recipe step.

    Returns:
        The absolute path of the step output on the local filesystem, which may or may
        not exist.
    """
    execution_dir_path = get_or_create_base_execution_directory(recipe_root_path=recipe_root_path)
    step_outputs_path = _get_step_output_directory_path(
        execution_directory_path=execution_dir_path,
        step_name=step_name,
    )
    return os.path.abspath(os.path.join(step_outputs_path, relative_path))


def get_state_output_dir(step_path: str, state_file_name: str) -> str:
    return os.path.join(step_path, state_file_name)


def get_step_component_output_path(step_path: str, component_name: str, extension='.csv') -> str:
    return os.path.join(step_path, hashlib.sha256(component_name.encode()).hexdigest() + extension)


def get_or_create_execution_directory(recipe_steps) -> str:
    """
    Obtains the path of the execution directory on the local filesystem corresponding to the
    specified recipe, creating the execution directory and its required contents if they do
    not already exist.
    Args:

        recipe_steps: A list of all the steps contained in the specified recipe.
    Returns:
        The absolute path of the execution directory on the local filesystem for the specified
        recipe.
    """
    if len(recipe_steps) == 0:
        raise ValueError('No steps provided')
    else:
        recipe_root_path = recipe_steps[0].context.recipe_root_path
        execution_dir_path = get_or_create_base_execution_directory(recipe_root_path)
        for step in recipe_steps:
            step_output_subdir_path = _get_step_output_directory_path(execution_dir_path, step.name)
            os.makedirs(step_output_subdir_path, exist_ok=True)
        return execution_dir_path


def get_step_fn(conf: Any, suffix: str) -> str:
    for attr, value in vars(conf).items():
        if isinstance(value, str) and value.endswith(suffix):
            return value

    raise MlflowException(
        f"{conf.__class__.__name__} should contain one attribute for the step function",
        error_code=MLFlowErrorCode.INVALID_PARAMETER_VALUE,
    )


def get_score_class(score: ScoreType) -> Type[Score]:
    return load_class(f'{SCORES_PATH}.{score.name}')


def get_features_target(dataset: Dataset, target_col: str) -> Tuple[Dataset, Dataset]:
    X: Dataset = dataset.select([c for c in dataset.columns if c != target_col])
    y: Dataset = dataset.select([target_col])
    return X, y


def resolve_dataset_source(conf: SourceConfig) -> DatasetSourceWrapper:
    return DatasetSourceWrapper.load_from_path(SOURCE_TO_MODULE[conf.type])(**conf.get_config.model_dump())
