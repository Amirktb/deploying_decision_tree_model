import math

import numpy as np

from decision_tree_model.predict import make_predictions

def test_make_prediction(input_data):
    # Given
    expected_first_prediction_value = 50.09
    expected_no_predictions = 17375

    # When
    result = make_predictions(input_data=input_data)

    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], np.float64)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    assert math.isclose(predictions[0], expected_first_prediction_value, abs_tol=100)
