import numpy as np
from config.core import config
from pipeline import rental_bikes_pipe
from processing.data_manager import load_dataset, save_pipeline
from sklearn.model_selection import train_test_split


def run_training() -> None:
    """
    Train the decision tree regressor model.
    """

    # load training data
    data = load_dataset(filename=config.app_config.data_file)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
    data[config.model_config.features],
    data[config.model_config.target],
    test_size=config.model_config.test_size,
    # random seed for reproducibility 
    random_state=config.model_config.random_state
    )
    y_train = np.log(y_train)

    # fitting the model with pipeline
    rental_bikes_pipe.fit(X_train, y_train)

    # saving the pipeline
    save_pipeline(pipeline_to_save=rental_bikes_pipe)


if __name__ == "__main__":
    run_training()
    