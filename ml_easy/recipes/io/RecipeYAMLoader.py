import json
import logging
import os
import posixpath
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import yaml

from ml_easy.recipes.constants import (
    ENCODING,
    RECIPE_CONFIG_FILE_NAME,
    RECIPE_PROFILE_DIR,
)

_logger = logging.getLogger(__name__)


class YamlLoader(ABC):

    @abstractmethod
    def read(self) -> str:
        pass

    def as_dict(self) -> Dict[str, Any]:
        return yaml.safe_load(self.read())


class RecipeYAMLoader(YamlLoader):

    def __init__(self, recipe_root_path: str, profile: Optional[str] = None):
        self._recipe_root_path = recipe_root_path
        self._profile = profile

    class UniqueKeyLoader(yaml.CSafeLoader):
        def construct_mapping(self, node, deep=False):
            mapping = set()
            for key_node, _ in node.value:
                key = self.construct_object(key_node, deep=deep)
                if key in mapping:
                    raise ValueError(f"Duplicate '{key}' key found in YAML.")
                mapping.add(key)
            return super().construct_mapping(node, deep)

    def render_and_merge_yaml(self) -> str:
        template_name = RECIPE_CONFIG_FILE_NAME
        context_name = posixpath.join(RECIPE_PROFILE_DIR, f"{self._profile}.yaml")
        from jinja2 import FileSystemLoader, StrictUndefined
        from jinja2.sandbox import SandboxedEnvironment

        template_path = os.path.join(self._recipe_root_path, template_name)
        context_path = os.path.join(self._recipe_root_path, context_name)

        j2_env = SandboxedEnvironment(
            loader=FileSystemLoader(self._recipe_root_path, encoding=ENCODING),
            undefined=StrictUndefined,
            line_comment_prefix='#',
        )

        def from_json(input_var):
            with open(input_var, encoding=ENCODING) as f:
                return json.load(f)

        j2_env.filters['from_json'] = from_json
        # Compute final source of context file (e.g. my-profile.yml), applying Jinja filters
        # like from_json as needed to load context information from files, then load into a dict
        context = j2_env.get_template(context_name).render({})
        context_dict = yaml.load(context, Loader=RecipeYAMLoader.UniqueKeyLoader) or {}

        # Substitute parameters from context dict into template
        source = j2_env.get_template(template_name).render(context_dict)
        return source

    def read(self) -> str:
        try:
            if self._profile:
                return self.render_and_merge_yaml()
            else:
                recipe_file_name = os.path.join(self._recipe_root_path, RECIPE_CONFIG_FILE_NAME)
                return open(recipe_file_name, 'r').read()
        except Exception as e:
            _logger.error('Failed to get recipe config', exc_info=e)
            raise
