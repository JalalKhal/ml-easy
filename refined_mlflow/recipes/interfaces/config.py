from abc import abstractmethod

from pydantic import BaseModel


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

    @property
    @abstractmethod
    def get_steps(self) -> BaseStepsConfig:
        pass


class BaseCard(BaseModel):
    step_output_path: str
