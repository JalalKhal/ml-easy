from typing import Optional

from pydantic import BaseModel



class RecipePathsConfig(BaseModel):
    recipe_root_path: str
    profile: Optional[str] = None

class BaseRecipeConfig(BaseModel):
    recipe: str


class BaseStepConfig(BaseModel):
    recipe_config: BaseRecipeConfig
    paths_config: RecipePathsConfig

class BaseCard(BaseModel):
    output_directory: str
