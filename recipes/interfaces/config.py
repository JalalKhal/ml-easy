
from pydantic import BaseModel



class BaseStepConfig(BaseModel):
    pass


class BaseStepsConfig(BaseModel):
    pass

class Context(BaseModel):
    recipe_root_path: str
    target_col: str

class BaseRecipeConfig(BaseModel):
    recipe: str
    context: Context

class BaseCard(BaseModel):
    step_output_path: str

