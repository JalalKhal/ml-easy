from ml_easy.recipes.interfaces.recipe import RecipeFactory
from ml_easy.recipes.steps.steps_config import RecipePathsConfig

if __name__ == '__main__':

    paths_config = RecipePathsConfig(
        recipe_root_path='/home/khaldi/Documents/github_repos/ml-easy/avs/fail_nff', profile='local'
    )
    recipe = RecipeFactory.create_recipe(paths_config)
    print(recipe.run())
