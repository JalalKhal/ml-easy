import abc
import logging
from typing import List, Dict, Any, Generic, TypeVar

from recipes.enum import MLFlowErrorCode
from recipes.exceptions import MlflowException
from recipes.io.RecipeYAMLoader import YamlLoader, RecipeYAMLoader
from recipes.recipes.config import RecipePathsConfig
from recipes.steps import BaseStep
from recipes.utils import get_recipe_name, _get_class_from_string

_logger = logging.getLogger(__name__)


U = TypeVar('U', bound="RecipeConfig")


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
        for field, value in conf:
            setattr(self, field, value)

        #self.steps: List[BaseStep] = self._resolve_recipe_steps()

    def _resolve_recipe_steps(self) -> List[BaseStep]:
        """
        Constructs and returns all recipe step objects from the recipe configuration.
        """
        return [
            s.from_recipe_config(self._config, self._recipe_root_path)
            for s in self._get_step_classes()
        ]

    @abc.abstractmethod
    def _get_step_classes(self):
        """
        Returns a list of step classes defined in the recipe.

        Concrete recipe class should implement this method.
        """
        pass


class Recipe:
    """
    A factory class that creates an instance of a recipe for a particular ML problem
    (e.g. regression, classification) or MLOps task (e.g. batch scoring) based on the current
    working directory and supplied configuration.

    .. code-block:: python
        :caption: Example

        import os
        from mlflow.recipes import Recipe

        os.chdir("~/recipes-regression-template")
        regression_recipe = Recipe(profile="local")
        regression_recipe.run(step="train")
    """

    def __new__(cls, recipe_paths_config: RecipePathsConfig) -> Any:
        """
        Creates an instance of an MLflow Recipe for a particular ML problem or MLOps task based
        on the current working directory and supplied configuration. The current working directory
        must be the root directory of an MLflow Recipe repository or a subdirectory of an
        MLflow Recipe repository.

        Args:
            profile: The name of the profile to use for configuring the problem-specific or
                task-specific recipe. Profiles customize the configuration of
                one or more recipe steps, and recipe executions with different profiles
                often produce different results.

        Returns:
            A recipe for a particular ML problem or MLOps task. For example, an instance of
            `RegressionRecipe <https://github.com/mlflow/recipes-regression-template>`_
            for regression problems.

        .. code-block:: python

            import os
            from mlflow.recipes import Recipe

            os.chdir("~/recipes-regression-template")
            regression_recipe = Recipe(profile="local")
            regression_recipe.run(step="train")
        """

        config = cls.read_config(recipe_paths_config)
        recipe = config.recipe
        recipe_path = recipe.replace("/", ".").replace("@", ".")
        class_name = f"recipes.{recipe_path}.RecipeImpl"
        recipe_class_module = cls.load_class(class_name)
        recipe_name = get_recipe_name(recipe_paths_config.recipe_root_path)
        _logger.info(f"Creating MLflow Recipe '{recipe_name}' with profile: '{recipe_paths_config.profile}'")
        return recipe_class_module(config)

    @classmethod
    def load_class(cls, class_name: str) -> Any:
        try:
            class_module = _get_class_from_string(class_name)
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
        reader: YamlLoader = RecipeYAMLoader(recipe_paths_config.recipe_root_path,
                                             recipe_paths_config.profile)
        config: Dict[str, Any] = reader.as_dict()
        recipe: str = config["recipe"]
        recipe_path: str = recipe.replace("/", ".").replace("@", ".")
        conf_class_name: str = f"recipes.{recipe_path}.ConfigImpl"
        conf_class_module = cls.load_class(conf_class_name)
        return conf_class_module.model_validate(config)
