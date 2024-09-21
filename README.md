# ML Easy

ML Easy is a fork of the MLflow Recipes framework, designed to streamline the process of developing, training, and deploying machine learning models. This framework provides a modular and customizable approach to building ML workflows.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Concepts](#key-concepts)
- [Configuration](#configuration)
- [Extensibility](#extensibility)

## Features

- **Modular Recipe-based Workflow**: Easily create and customize ML workflows using a recipe-based approach.
- **Flexible Step Configuration**: Configure each step of your ML pipeline with custom YAML files.
- **Automatic Execution State Tracking**: Keep track of the execution status for each step in your workflow.
- **Support for Various ML Tasks**: Built-in support for common ML tasks.
- **MLflow Integration**: Seamless integration with MLflow for experiment tracking and model management.
- **Extensible Architecture**: Easily add custom steps and extend the framework to suit your specific needs.

## Installation

To install ML Easy Recipes, clone the repository and install the required dependencies:

```bash
pip install ml-easy
```
## Usage
Here's a basic example of how to use ML Easy Recipes:
``` python
from ml_easy.recipes import RecipeFactory
from ml_easy.recipes.steps.steps_config import RecipePathsConfig

# Configure the recipe paths
recipe_paths_config = RecipePathsConfig(
    recipe_root_path="path/to/your/recipe",
    profile="your_profile"
)

# Create and run the recipe
recipe = RecipeFactory.create_recipe(recipe_paths_config)
recipe.run()
```
## Project Structure

The framework is organized into several key components:

- `BaseRecipe`: The core class representing an ML workflow.
- `BaseStep`: Abstract base class for individual steps in the workflow.
- `RecipeFactory`: Factory class for creating recipe instances.
- `StepExecutionState`: Tracks the execution state of each step.
- `Context`: Holds context information for the entire recipe.
- `BaseStepConfig`: Base class for step configurations.
- `BaseCard`: Base class for step result cards.

## Key Concepts
- **Steps**: Individual ML operations, such as data ingestion, model training, or evaluation. Each step is implemented as a subclass of `BaseStep`.
- **Recipes**: Ordered compositions of Steps used to solve an ML problem or perform an MLOps task. Implemented as subclasses of `BaseRecipe`.
- **Templates**: Standardized, modular layouts containing customizable code and configurations for a Recipe.
- **Profiles**: User-specific or environment-specific configurations for a Recipe.
- **Step Cards**: Displays of results produced by running a Step, including dataset profiles and model performance.
- **Execution State**: Tracked using the `StepExecutionState` class, which includes status, timestamp, and stack trace information.

## Configuration
The framework uses Pydantic models for configuration:

- `BaseRecipeConfig`: Base configuration for recipes.
- `BaseStepConfig`: Base configuration for individual steps.
- `Context`: Holds context information like recipe root path, target column, and experiment details.

## Extensibility
You can extend the framework by:

1. Creating new step classes that inherit from `BaseStep`.
2. Implementing custom recipe classes that inherit from `BaseRecipe`.
3. Defining new configuration classes for steps and recipes.

