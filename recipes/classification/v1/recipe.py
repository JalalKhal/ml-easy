from typing import Type, Dict

from recipes.classification.v1.config import ClassificationRecipeConfig
from recipes.classification.v1.steps import ClassificationIngestStep
from recipes.interfaces.recipe import BaseRecipe
from recipes.interfaces.step import BaseStep


class ClassificationRecipe(BaseRecipe[ClassificationRecipeConfig]):
    _RECIPE_STEPS = {
        'ingest' : ClassificationIngestStep
    }

    @property
    def recipe_steps(self) -> Dict[str, Type[BaseStep]]:
        return self._RECIPE_STEPS







