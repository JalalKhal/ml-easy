from recipes.classification.v1.config import ClassificationRecipeConfig
from recipes.recipes.recipe import BaseRecipe


class ClassificationRecipe(BaseRecipe[ClassificationRecipeConfig]):
    _RECIPE_STEPS = (

    )

    def _get_step_classes(self):
        """
        Returns a list of step classes defined in the recipe.

        Concrete recipe class should implement this method.
        """
        pass
