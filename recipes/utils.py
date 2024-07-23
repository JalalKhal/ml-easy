import codecs
import importlib
import json
import logging
import os
import pathlib
import posixpath
from typing import Optional, Dict, Any

from yaml import CSafeLoader as YamlSafeLoader
import yaml

ENCODING = "utf-8"

_RECIPE_CONFIG_FILE_NAME = "recipe.yaml"
_RECIPE_PROFILE_DIR = "profiles"
_logger = logging.getLogger(__name__)


class UniqueKeyLoader(YamlSafeLoader):
    def construct_mapping(self, node, deep=False):
        mapping = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"Duplicate '{key}' key found in YAML.")
            mapping.add(key)
        return super().construct_mapping(node, deep)


def merge_dicts(dict_a, dict_b, raise_on_duplicates=True):
    """This function takes two dictionaries and returns one singular merged dictionary.

    Args:
        dict_a: The first dictionary.
        dict_b: The second dictionary.
        raise_on_duplicates: If True, the function raises ValueError if there are duplicate keys.
            Otherwise, duplicate keys in `dict_b` will override the ones in `dict_a`.

    Returns:
        A merged dictionary.
    """
    duplicate_keys = dict_a.keys() & dict_b.keys()
    if raise_on_duplicates and len(duplicate_keys) > 0:
        raise ValueError(f"The two merging dictionaries contain duplicate keys: {duplicate_keys}.")
    return {**dict_a, **dict_b}


def read_yaml(root, file_name):
    """Read data from yaml file and return as dictionary

    Args:
        root: Directory name.
        file_name: File name. Expects to have '.yaml' extension.

    Returns:
        Data in yaml file as dictionary.
    """
    file_path = os.path.join(root, file_name)
    with codecs.open(file_path, mode="r", encoding=ENCODING) as yaml_file:
        return yaml.load(yaml_file, Loader=YamlSafeLoader)


def render_and_merge_yaml(root, template_name, context_name):
    """Renders a Jinja2-templated YAML file based on a YAML context file, merge them, and return
    result as a dictionary.

    Args:
        root: Root directory of the YAML files.
        template_name: Name of the template file.
        context_name: Name of the context file.

    Returns:
        Data in yaml file as dictionary.
    """
    from jinja2 import FileSystemLoader, StrictUndefined
    from jinja2.sandbox import SandboxedEnvironment

    template_path = os.path.join(root, template_name)
    context_path = os.path.join(root, context_name)

    j2_env = SandboxedEnvironment(
        loader=FileSystemLoader(root, encoding=ENCODING),
        undefined=StrictUndefined,
        line_comment_prefix="#",
    )

    def from_json(input_var):
        with open(input_var, encoding="utf-8") as f:
            return json.load(f)

    j2_env.filters["from_json"] = from_json
    # Compute final source of context file (e.g. my-profile.yml), applying Jinja filters
    # like from_json as needed to load context information from files, then load into a dict
    context_source = j2_env.get_template(context_name).render({})
    context_dict = yaml.load(context_source, Loader=UniqueKeyLoader) or {}

    # Substitute parameters from context dict into template
    source = j2_env.get_template(template_name).render(context_dict)
    rendered_template_dict = yaml.load(source, Loader=UniqueKeyLoader)
    return rendered_template_dict

def get_recipe_name(recipe_root_path: Optional[str] = None) -> str:
    """
    Obtains the name of the specified recipe or of the recipe corresponding to the current
    working directory.

    Args:
        recipe_root_path: The absolute path of the recipe root directory on the local
            filesystem. If unspecified, the recipe root directory is resolved from the current
            working directory.

    Raises:
        MlflowException: If the specified ``recipe_root_path`` is not a recipe root
            directory or if ``recipe_root_path`` is ``None`` and the current working directory
            does not correspond to a recipe.

    Returns:
        The name of the specified recipe.
    """
    return os.path.basename(recipe_root_path)


def get_recipe_root_path() -> str:
    """
    Obtains the path of the recipe corresponding to the current working directory,
    Returns:
        The absolute path of the recipe root directory on the local filesystem.
    """
    # In the release version of MLflow Recipes, each recipe will be its own git repository.
    # To improve developer velocity for now, we choose to treat a recipe as a directory, which
    # may be a subdirectory of a git repo. The logic for resolving the repository root for
    # development purposes finds the first `recipe.yaml` file by traversing up the directory
    # tree, while the release version will find the recipe repository root (commented out below)
    curr_dir_path = pathlib.Path.cwd()

    while True:
        recipe_yaml_path_to_check = curr_dir_path / _RECIPE_CONFIG_FILE_NAME
        if recipe_yaml_path_to_check.exists():
            return str(curr_dir_path.resolve())
        elif curr_dir_path != curr_dir_path.parent:
            curr_dir_path = curr_dir_path.parent
        else:
            # If curr_dir_path == curr_dir_path.parent,
            # we have reached the root directory without finding
            # the desired recipe.yaml file
            raise Exception(f"Failed to find {_RECIPE_CONFIG_FILE_NAME}!")


def get_recipe_config(
        recipe_root_path: Optional[str] = None, profile: Optional[str] = None
) -> Dict[str, Any]:
    """
    Obtains a dictionary representation of the configuration for the specified recipe.

    Args:
        recipe_root_path: The absolute path of the recipe root directory on the local
            filesystem. If unspecified, the recipe root directory is resolved from the current
            working directory.
        profile: The name of the profile under the `profiles` directory to use, e.g. "dev" to
            use configs from "profiles/dev.yaml".

    Raises:
        MlflowException: If the specified ``recipe_root_path`` is not a recipe root directory
            or if ``recipe_root_path`` is ``None`` and the current working directory does not
            correspond to a recipe.

    Returns:
        The configuration of the specified recipe.
    """
    recipe_root_path = recipe_root_path or get_recipe_root_path()
    try:
        if profile:
            # Jinja expects template names in posixpath format relative to environment root,
            # so use posixpath to construct the relative path here.
            profile_relpath = posixpath.join(_RECIPE_PROFILE_DIR, f"{profile}.yaml")
            profile_file_path = os.path.join(
                recipe_root_path, _RECIPE_PROFILE_DIR, f"{profile}.yaml"
            )
            return render_and_merge_yaml(
                recipe_root_path, _RECIPE_CONFIG_FILE_NAME, profile_relpath
            )
        else:
            return read_yaml(recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
    except Exception as e:
        _logger.error("Failed to get recipe config", exc_info=e)
        raise


def _get_class_from_string(fully_qualified_class_name):
    module, class_name = fully_qualified_class_name.rsplit(".", maxsplit=1)
    return getattr(importlib.import_module(module), class_name)
