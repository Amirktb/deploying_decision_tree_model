from feature_engine.encoding import OneHotEncoder
from feature_engine.transformation import YeoJohnsonTransformer
from feature_engine.selection import DropCorrelatedFeatures, DropFeatures
from sklearn.tree import DecisionTreeRegressor

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from config.core import config

rental_bikes_pipe = Pipeline(
    [
        # ==== VARIABLE TRANSFORMATION =====
        # yeo johnson transformation
        (
            'yeojohnson', 
            YeoJohnsonTransformer(
                variables=config.model_config.numerical_vars_yj
            ),
        ),
        # ==== CATEGORICAL ENCODING =====
        (
            'onehot_encoder', 
            OneHotEncoder(
                variables=config.model_config.categorical_vars
                ),
        ),
        # ==== DROP CORRELATED FEATURES =====
        (
        'drop_features', 
        DropFeatures(
            features_to_drop=config.model_config.features_to_drop
            ),
        ),
        (
            'drop_correlated_features', 
            DropCorrelatedFeatures(
                method='pearson', 
                threshold=0.8
                ),
        ),
        # ==== MINMAX SCALER =====
        (
            'scaler', 
            MinMaxScaler()
        ),
        # ==== DECISION TREE REGRESSOR =====
        (
            'DecisionTreeRegressor', 
            DecisionTreeRegressor(
                max_depth=config.model_config.max_depth,
                max_leaf_nodes=config.model_config.max_leaf_nodes,
                min_samples_leaf=config.model_config.min_samples_split,
            ),
        ),
    ]
)
