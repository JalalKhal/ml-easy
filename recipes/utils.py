import codecs
import hashlib
import importlib
import json
import logging
import os
import pathlib
import posixpath
from typing import Optional, Dict, Any, List

from pydantic import BaseModel
from yaml import CSafeLoader as YamlSafeLoader
import yaml

from recipes.constants import STEPS_SUBDIRECTORY_NAME, STEP_OUTPUTS_SUBDIRECTORY_NAME
from recipes.enum import MLFlowErrorCode
from recipes.env_vars import MLFLOW_RECIPES_EXECUTION_DIRECTORY
from recipes.exceptions import MlflowException


def get_recipe_name(recipe_root_path: Optional[str] = None) -> str:
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

def _get_class_from_string(fully_qualified_class_name):
    module, class_name = fully_qualified_class_name.rsplit(".", maxsplit=1)
    return getattr(importlib.import_module(module), class_name)

def load_config(obj: Any, config: Any):
    for field, value in config.__dict__.items():
        setattr(obj, field, value)



def _get_execution_directory_basename(recipe_root_path):
    """
    Obtains the basename of the execution directory corresponding to the specified recipe.

    Args:
        recipe_root_path: The absolute path of the recipe root directory on the local
            filesystem.

    Returns:
        The basename of the execution directory corresponding to the specified recipe.
    """
    return hashlib.sha256(os.path.abspath(recipe_root_path).encode("utf-8")).hexdigest()

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
    execution_directory_basename = _get_execution_directory_basename(
        recipe_root_path=recipe_root_path
    )

    execution_dir_path = os.path.abspath(
        MLFLOW_RECIPES_EXECUTION_DIRECTORY.get()
        or os.path.join(os.path.expanduser("~"), ".mlflow", "recipes", execution_directory_basename)
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

def get_step_output_path(recipe_root_path: str, step_name: str) -> str:
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
    return os.path.abspath(os.path.join(step_outputs_path))


def get_state_output_dir(step_path: str, state_file_name: str) -> str:
    return os.path.join(step_path, state_file_name)

def get_step_component_output_path(step_path: str, component_name: str, extension = ".csv") -> str:
    return os.path.join(step_path,
                        hashlib.sha256(component_name.encode()).hexdigest() + extension)


def _get_or_create_execution_directory(recipe_steps) -> str:
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
        raise ValueError("No steps provided")
    else:
        recipe_root_path = recipe_steps[0].context.recipe_root_path
        execution_dir_path = get_or_create_base_execution_directory(recipe_root_path)
        for step in recipe_steps:
            step_output_subdir_path = _get_step_output_directory_path(execution_dir_path, step.name)
            os.makedirs(step_output_subdir_path, exist_ok=True)
        return execution_dir_path