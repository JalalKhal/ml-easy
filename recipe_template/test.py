from recipes.interfaces.recipe import Recipe
from recipes.steps.cards_config import StepMessage

if __name__ == '__main__':
    recipe = Recipe("/home/khaldi/Documents/github_repos/refined_mlflow/recipe_template")
    message: StepMessage = recipe.run()
