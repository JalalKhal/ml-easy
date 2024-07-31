from recipes.interfaces.recipe import Recipe
from recipes.steps.steps_config import RecipePathsConfig

if __name__ == '__main__':
    paths_config = RecipePathsConfig(
        recipe_root_path="/home/khaldi/Documents/github_repos/refined_mlflow/recipe_template", profile="local")
    recipe = Recipe.create_recipe(paths_config)

    print(recipe.run())
