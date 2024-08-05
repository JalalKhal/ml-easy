from recipes.enum import FilterType

STEPS_SUBDIRECTORY_NAME = 'steps'
STEP_OUTPUTS_SUBDIRECTORY_NAME = 'outputs'
EXT_PY = '.py'
SCORES_PATH = 'recipes.steps.evaluate.score'
ENCODING = 'utf-8'
RECIPE_CONFIG_FILE_NAME = 'recipe.yaml'
RECIPE_PROFILE_DIR = 'profiles'

FILTER_TO_MODULE = {
    FilterType['EQUAL']: 'recipes.steps.transform.filters.EqualFilter',
    FilterType['IN']: 'recipes.steps.transform.filters.InFilter',
}
