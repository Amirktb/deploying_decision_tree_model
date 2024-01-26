from typing import Union, Dict, Any

import numpy as np
import pandas as pd

from decision_tree_model import __version__ as _version
from decision_tree_model.config.core import config
from decision_tree_model.processing.data_manager import load_pipeline
from decision_tree_model.processing.validation import validate_inputs

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_rental_bikes_pipe = load_pipeline(file_name=pipeline_file_name)


def make_predictions(
        *,
        input_data: Union[str, None, Dict[Any, Any]],
        )
    """Make predictions using the saved model pipeline."""

    data = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input_data=data)
    results = {"predictions": None, "version": _version, "errors": errors}

    if not errors:
        predictions = _rental_bikes_pipe.predict(
            X=validated_data[config.model_config.features]
        )
        results = {
            "predictions": [np.exp(pred) for pred in predictions],
            "version": _version,
            "errors": errors,
        }

    return results
