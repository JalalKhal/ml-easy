import posixpath
from abc import ABC, abstractmethod
import json
import logging
import os
from typing import Optional, Dict, Any

import yaml

from recipes.constants import RECIPE_CONFIG_FILE_NAME

_logger = logging.getLogger(__name__)


class YamlLoader(ABC):

    @abstractmethod
    def read(self) -> str:
        pass

    def as_dict(self) -> Dict[str, Any]:
        return yaml.safe_load(self.read())


class RecipeYAMLoader(YamlLoader):

    def __init__(self,
                 recipe_root_path: str):
        self._recipe_root_path = recipe_root_path

    class UniqueKeyLoader(yaml.CSafeLoader):
        def construct_mapping(self, node, deep=False):
            mapping = set()
            for key_node, _ in node.value:
                key = self.construct_object(key_node, deep=deep)
                if key in mapping:
                    raise ValueError(f"Duplicate '{key}' key found in YAML.")
                mapping.add(key)
            return super().construct_mapping(node, deep)

    def read(self) -> str:
        try:
            recipe_file_name = os.path.join(self._recipe_root_path, RECIPE_CONFIG_FILE_NAME)
            return open(recipe_file_name, "r").read()
        except Exception as e:
            _logger.error("Failed to get recipe config", exc_info=e)
            raise
