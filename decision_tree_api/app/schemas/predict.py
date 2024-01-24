from typing import Any, List, Optional

from pydantic import BaseModel
from decision_tree_model.processing.validation import RentalBikesInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[float]]


class MultipleRentalBikesInputs(BaseModel):
    inputs: List[RentalBikesInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "cnt_lag1": 16,
                        "cnt_lag2": 40,
                        "season": 1,
                        "yr": 0,
                        "mnth": 1,
                        "hr": 0,
                        "holiday": 0,
                        "weekday": 6,
                        "workingday": 0,
                        "weathersit": 1,
                        "temp": 0.24,
                        "atemp": 0.2879,
                        "hum": 0.81,
                        "windspeed": 0,
                        "casual": 3,
                        "registered": 13,
                    }
                ]
            }
        }
        