import pytest

from decision_tree_model.config.core import config
from decision_tree_model.processing.data_manager import load_dataset


@pytest.fixture
def input_data():
    return load_dataset(filename=config.app_config.data_file)
