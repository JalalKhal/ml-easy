from typing import Optional

from pydantic import BaseModel



class RecipePathsConfig(BaseModel):
    recipe_root_path: str
    profile: Optional[str] = None

class RecipeConfig(BaseModel):
    recipe: str