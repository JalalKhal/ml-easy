from typing import Union

from pydantic import BaseModel

from recipes.classification.v1.config import ClassificationRecipeConfig


class BaseStepConfig(BaseModel):
    pass


class BaseStepsConfig(BaseModel):
    pass

class Context(BaseModel):
    recipe_root_path: str
    target_col: str

class BaseRecipeConfig(BaseModel):
    steps: Union["ClassificationRecipeConfig"]
    recipe: str
    context: Context

class BaseCard(BaseModel):
    step_output_path: str

