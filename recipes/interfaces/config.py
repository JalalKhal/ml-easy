
from pydantic import BaseModel



class BaseStepConfig(BaseModel):
    pass


class BaseStepsConfig(BaseModel):
    pass

class BaseRecipeConfig(BaseModel):
    recipe_root_path: str
    recipe: str
    steps: BaseStepsConfig


class Context(BaseModel):
    recipe_root_path: str

class BaseCard(BaseModel):
    step_output_path: str

