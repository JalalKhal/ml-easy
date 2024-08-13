import abc
import logging
from typing import Any, Dict, Generic, List, Type, TypeVar

from ml_easy.recipes.enum import MLFlowErrorCode
from ml_easy.recipes.exceptions import MlflowException
from ml_easy.recipes.interfaces.config import BaseRecipeConfig, BaseStepConfig
from ml_easy.recipes.interfaces.step import BaseStep
from ml_easy.recipes.io.RecipeYAMLoader import RecipeYAMLoader, YamlLoader
from ml_easy.recipes.steps.cards_config import StepMessage
from ml_easy.recipes.steps.steps_config import RecipePathsConfig
from ml_easy.recipes.utils import (
    get_class_from_string,
    get_or_create_execution_directory,
    get_recipe_name,
)

_logger = logging.getLogger(__name__)


U = TypeVar('U', bound='BaseRecipeConfig')


class BaseRecipe(abc.ABC, Generic[U]):
    """
    Base Recipe
    """

    def __init__(self, conf: U) -> None:
        """
        Recipe base class.

        Args:
            recipe_root_path: String path to the directory under which the recipe template
                such as recipe.yaml, profiles/{profile}.yaml and steps/{step_name}.py are defined.
            profile: String specifying the profile name, with which
                {recipe_root_path}/profiles/{profile}.yaml is read and merged with
                recipe.yaml to generate the configuration to run the recipe.
        """
        self._conf: U = conf
        self.steps: List[BaseStep] = self._resolve_recipe_steps()

    def _resolve_recipe_steps(self) -> List[BaseStep]:
        steps: List[BaseStep] = []
        for step_name in self._conf.get_steps.model_fields.keys():
            step_class: Type[BaseStep] = self.recipe_steps[step_name]
            step_config: BaseStepConfig = getattr(self._conf.get_steps, step_name)
            steps.append(step_class(step_config, self._conf.context))
        return steps

    @property
    @abc.abstractmethod
    def recipe_steps(self) -> Dict[str, Type[BaseStep]]:
        pass

    def run(self) -> StepMessage:
        """
        Run the entire recipe if a step is not specified.
        Args:
        Returns:
            None
        """
        message = StepMessage()
        get_or_create_execution_directory(self.steps)
        for step in self.steps:
            message = step.run(message)
        return message


class RecipeFactory:
    """
    A factory class that creates an instance of a recipe for a particular ML problem
    (e.g. regression, classification) or MLOps task (e.g. batch scoring) based on the current
    working directory and supplied configuration.

    .. code-block:: python
        :caption: Example


    """

    @classmethod
    def create_recipe(cls, recipe_paths_config: RecipePathsConfig) -> BaseRecipe:
        """
        Creates an instance of an MLflow Recipe for a particular ML problem or MLOps task based
        on the current working directory and supplied configuration. The current working directory
        must be the root directory of an MLflow Recipe repository or a subdirectory of an
        MLflow Recipe repository.
        """

        config = cls.read_config(recipe_paths_config)
        recipe = config.recipe
        recipe_path = recipe.replace('/', '.').replace('@', '.')
        class_name = f"ml_easy.recipes.{recipe_path}.RecipeImpl"
        recipe_class_module = cls.load_class(class_name)
        recipe_name = get_recipe_name(recipe_paths_config.recipe_root_path)
        _logger.info(f"Creating MLflow Recipe '{recipe_name}' with profile: '{recipe_paths_config.profile}'")
        return recipe_class_module(config)

    @classmethod
    def load_class(cls, class_name: str) -> Any:
        try:
            class_module = get_class_from_string(class_name)
        except Exception as e:
            if isinstance(e, ModuleNotFoundError):
                raise MlflowException(
                    f"Failed to find {class_name}.",
                    error_code=MLFlowErrorCode.INVALID_PARAMETER_VALUE,
                ) from None
            else:
                raise MlflowException(
                    f"Failed to construct {class_name}. Error: {e!r}",
                    error_code=MLFlowErrorCode.INVALID_PARAMETER_VALUE,
                ) from None
        return class_module

    @classmethod
    def read_config(cls, recipe_paths_config: RecipePathsConfig) -> Any:
        reader: YamlLoader = RecipeYAMLoader(recipe_paths_config.recipe_root_path, recipe_paths_config.profile)
        config: Dict[str, Any] = reader.as_dict()
        recipe: str = config['recipe']
        recipe_path: str = recipe.replace('/', '.').replace('@', '.')
        conf_class_name: str = f"ml_easy.recipes.{recipe_path}.ConfigImpl"
        conf_class_module = cls.load_class(conf_class_name)
        return conf_class_module.model_validate(config)
