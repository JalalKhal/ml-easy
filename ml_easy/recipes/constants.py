from ml_easy.recipes.enum import FilterType, SourceType

STEPS_SUBDIRECTORY_NAME = 'steps'
STEP_OUTPUTS_SUBDIRECTORY_NAME = 'outputs'
EXT_PY = '.py'
SCORES_PATH = 'ml_easy.recipes.steps.evaluate.score'
ENCODING = 'utf-8'
RECIPE_CONFIG_FILE_NAME = 'recipe.yaml'
RECIPE_PROFILE_DIR = 'profiles'
EXECUTION_STATE_FILE_NAME = 'execution_state.json'
CUSTOM_STEPS_DIR = 'steps'
SUFFIX_FN = '_fn'

FILTER_TO_MODULE = {
    FilterType['EQUAL']: 'ml_easy.recipes.steps.transform.filters.EqualFilter',
    FilterType['IN']: 'ml_easy.recipes.steps.transform.filters.InFilter',
}

SOURCE_TO_MODULE = {
    SourceType[
        'SQL_ALCHEMY_BASED'
    ]: 'ml_easy.recipes.steps.register.mlflow_source.sql_table_dataset_source.SQLTableDatasetSource'
}
