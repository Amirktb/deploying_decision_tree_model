from pathlib import Path
from typing import List

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from decision_tree_model import __version__ as _version
from decision_tree_model.config.core import config, DATASET_PATH, TRAINED_MODEL_DIR


def load_dataset(*, filename: str) -> pd.DataFrame:
    rental_bikes_df = pd.read_csv(Path(f"{DATASET_PATH}/{filename}"))
    # convert categorical columns to dtype category
    for var in config.model_config.categorical_vars:
        rental_bikes_df[f"{var}"] = rental_bikes_df[f"{var}"].astype('category')

    # add lag variables to the data
    rental_bikes_df['cnt_lag1'] = rental_bikes_df['cnt'].shift(-1)
    rental_bikes_df['cnt_lag2'] = rental_bikes_df['cnt'].shift(-2)
    rental_bikes_df.dropna(inplace=True)
    return rental_bikes_df


def save_pipeline(*, pipeline_to_save: Pipeline) -> None:
    """
    Save the pipeline.
    It saves  the versioned model and overwrites any previous
    saved models. Ensures that when the model is published, only
    one model is saved and we know how the model was built.
    """

    # versioned file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipeline(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_save, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    trained_model_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=trained_model_path)
    return trained_model


def remove_old_pipeline(*, files_to_keep: List[str]) -> None:
    """
    Removes old pipelines.
    This ensures that we have one-to-one mapping between the
    package version and version of the model, used by other
    applications.
    """

    do_not_delete = files_to_keep + ['__init__.py']
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
