from recipes.recipes.config import RecipePathsConfig
from recipes.recipes.recipe import Recipe

if __name__ == '__main__':
    paths_config = RecipePathsConfig(recipe_root_path = "/home/khaldi/Documents/github_repos/refined_mlflow/recipe_template", profile = "local")
    recipe = Recipe(paths_config)

    print()