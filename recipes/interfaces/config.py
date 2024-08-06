from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    pass


class BaseStepConfig(BaseModel):
    pass


class BaseStepsConfig(BaseModel):
    pass


class Experiment(BaseModel):
    product_name: str
    name: str
    tracking_uri: str


class Context(BaseModel):
    recipe_root_path: str
    target_col: str
    experiment: Experiment


class BaseRecipeConfig(BaseModel):
    recipe: str
    context: Context


class BaseCard(BaseModel):
    step_output_path: str
