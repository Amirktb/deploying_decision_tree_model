import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from loguru import logger
from regression_model import __version__ as model_version
from regression_model.predict import make_prediction

from decision_tree_api.app.schemas import __version__
from app.config import settings
from decision_tree_api.app import schemas

