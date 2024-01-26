from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from decision_tree_model.config.core import config


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    validated_data = input_data.copy()

    vars_with_na = [
        var for var in config.model_config.features
        if validated_data[var].isnull().sum() > 0
    ]
    validated_data.dropna(subset=vars_with_na, inplace=True)
    return validated_data


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """
    Check features of model input and for unprocessable values.
    """

    # drop unnecessary columns
    input_data.drop(['instant', 'dteday'],
                    axis=1,
                    inplace=True)
    # convert categorical columns to dtype category
    for var in config.model_config.categorical_vars:
        input_data[f"{var}"] = input_data[f"{var}"].astype('category')
    # add lag variables to the data
    input_data['cnt_lag1'] = input_data['cnt'].shift(-1)
    input_data['cnt_lag2'] = input_data['cnt'].shift(-2)
    input_data.dropna(inplace=True)
    relevant_data = input_data[config.model_config.features].copy()
    validated_data = drop_na_inputs(input_data=relevant_data)
    errors = None

    try:
        MultipleRentalBikesInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class RentalBikesInputSchema(BaseModel):
    cnt_lag1: Optional[int]
    cnt_lag2: Optional[int]
    season: Optional[int]
    yr: Optional[int]
    mnth: Optional[int]
    hr: Optional[int]
    holiday: Optional[int]
    weekday: Optional[int]
    workingday: Optional[int]
    weathersit: Optional[int]
    temp: Optional[float]
    atemp: Optional[float]
    hum: Optional[float]
    windspead: Optional[float]
    casual: Optional[int]
    registered: Optional[int]


class MultipleRentalBikesInputs(BaseModel):
    inputs: List[RentalBikesInputSchema]
