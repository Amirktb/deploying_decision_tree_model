import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from loguru import logger
from decision_tree_model import __version__ as model_version
from decision_tree_model.predict import make_prediction

from decision_tree_api.app.schemas import __version__
from app.config import settings
from decision_tree_api.app import schemas


# performing path operations
api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get.
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )


@api_router.get("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleRentalBikesInputs) -> Any:
    """
    Predicting number of rental bikes with 
    the decision tree model.
    """

    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))

    logger.info(f"Making prediction on inputs: {input_data.inputs}")
    results = make_prediction(input_data=input_df.replace({np.nan: None}))

    if results["errors"] is not None:
        logger.warning(f"prediction validation error: {results.get('errors')}")
        raise HTTPException(status_code=400, detail=json.loads(results['errors']))
    
    logger.info(f"Prediction results: {results.get('predictions')}")

    return results
