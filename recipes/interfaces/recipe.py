import abc
import logging
from typing import List, Dict, Any, Generic, TypeVar, Type

from recipes.interfaces.config import BaseStepConfig
from recipes.interfaces.step import BaseStep
from recipes.io.RecipeYAMLoader import YamlLoader, RecipeYAMLoader
from recipes.steps.cards_config import StepMessage
from recipes.utils import get_recipe_name, get_or_create_execution_directory, load_class

_logger = logging.getLogger(__name__)


U = TypeVar('U', bound="BaseRecipeConfig")


class BaseRecipe(abc.ABC, Generic[U]):
    """
    Base Recipe
    """

    def __init__(self, conf: U) -> None:
        """
        Recipe base class.

        Args:

        """
        self._conf:U = conf
        self.steps: List[BaseStep] = self._resolve_recipe_steps()

    def _resolve_recipe_steps(self) -> List[BaseStep]:
        steps: List[BaseStep] = []
        for step_name in self._conf.steps.__fields__.keys():
            step_class: Type[BaseStep] = self.recipe_steps[step_name]
            step_config: BaseStepConfig = getattr(self._conf.steps, step_name)
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


class Recipe:
    """
    A factory class that creates an instance of a recipe for a particular ML problem
    (e.g. regression, classification) or MLOps task (e.g. batch scoring) based on the current
    working directory and supplied configuration.

    .. code-block:: python
        :caption: Example


    """

    def __new__(cls, recipe_root_path: str) -> Any:
        config = cls.read_config(recipe_root_path)
        recipe = config.recipe
        recipe_path = recipe.replace("/", ".").replace("@", ".")
        class_name = f"recipes.{recipe_path}.RecipeImpl"
        recipe_class_module = load_class(class_name)
        recipe_name = get_recipe_name(recipe_root_path)
        _logger.info(f"Creating MLflow Recipe '{recipe_name}")
        return recipe_class_module(config)

    @classmethod
    def read_config(cls, recipe_root_path: str) -> Any:
        reader: YamlLoader = RecipeYAMLoader(recipe_root_path)
        config: Dict[str, Any] = reader.as_dict()
        recipe: str = config["recipe"]
        recipe_path: str = recipe.replace("/", ".").replace("@", ".")
        conf_class_name: str = f"recipes.{recipe_path}.ConfigImpl"
        conf_class_module = load_class(conf_class_name)
        return conf_class_module.model_validate(config)
