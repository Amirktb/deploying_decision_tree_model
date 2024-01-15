from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel
from strictyaml import YAML, load

import decision_tree_model

# package directories
PACKAGE_ROOT = Path(decision_tree_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_PATH = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


class AppConfig(BaseModel):
    """
    Application level config.
    """

    package_name: str
    data_file: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    """
    Configurations for feature engineering
    and model training.
    """

    target: str
    features: List[str]
    test_size: float
    random_state: int
    max_depth: int
    max_leaf_nodes: int
    min_samples_split: int
    numerical_vars_yj: List[str]
    categorical_vars: List[str]
    features_to_drop: List[str]


class Config(BaseModel):
    """
    Master configuration. a wrapper for
    both model and app configurations.
    """

    app_config: AppConfig
    model_config: ModelConfig


def find_config_file() -> Path:
    """
    Locate the configuration file.
    """
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"config file not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> YAML:
    """
    Fetching the configurations from yaml file.
    """

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as cfg_file:
            parsed_cfg_file = load(cfg_file.read())
            return parsed_cfg_file
    OSError(f"path to the configurations not found at {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """
    create and run validations on config values.
    """

    if not parsed_config:
        parsed_config = fetch_config_from_yaml()

    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        model_config=ModelConfig(**parsed_config.data)
    )

    return _config


config = create_and_validate_config()
