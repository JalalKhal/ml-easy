from typing import Dict, Type

from recipes.classification.v1.config import ClassificationRecipeConfig
from recipes.classification.v1.steps import (
    ClassificationEvaluateStep,
    ClassificationIngestStep,
    ClassificationRegisterStep,
    ClassificationSplitStep,
    ClassificationTrainStep,
    ClassificationTransformStep,
)
from recipes.interfaces.recipe import BaseRecipe
from recipes.interfaces.step import BaseStep


class ClassificationRecipe(BaseRecipe[ClassificationRecipeConfig]):
    _RECIPE_STEPS: Dict[str, Type[BaseStep]] = {
        'ingest': ClassificationIngestStep,
        'transform': ClassificationTransformStep,
        'split': ClassificationSplitStep,
        'train': ClassificationTrainStep,
        'evaluate': ClassificationEvaluateStep,
        'register_': ClassificationRegisterStep,
    }

    @property
    def recipe_steps(self) -> Dict[str, Type[BaseStep]]:
        return self._RECIPE_STEPS
