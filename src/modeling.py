import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_absolute_error

def create_xgb_model(params=None):
    """
    Create and return default parameters for an XGBoost model optimized for time series forecasting.
    Research suggests that a moderate learning rate, a maximum tree depth around 6, and regularization parameters
    can be effective. You can fine-tune these parameters further.
    """
    if params is None:
        params = {
            "objective": "reg:squarederror",  # using squared error objective
            "eval_metric": "mae",             # track MAE during training
            "eta": 0.1,                       # learning rate
            "max_depth": 6,                   # tree depth
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42
        }
    return params

def train_xgb_model(params, X_train, y_train, X_val, y_val, num_rounds=100):
    """
    Trains an XGBoost model using the provided parameters.
    Early stopping is applied with a window of 10 rounds.
    
    Args:
        params: Parameter dictionary for XGBoost.
        X_train, y_train: Training feature matrix and target vector.
        X_val, y_val: Validation feature matrix and target vector.
        num_rounds: Maximum number of boosting rounds.
    
    Returns:
        model: The trained XGBoost model.
    """
    # Create DMatrices required by xgb.train
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    eval_list = [(dval, "eval")]
    
    model = xgb.train(params, 
                      dtrain, 
                      num_boost_round=num_rounds, 
                      evals=eval_list, 
                      early_stopping_rounds=10, 
                      verbose_eval=True)
    return model

def predict_xgb_model(model, X):
    """
    Make predictions using a trained XGBoost model.
    """
    dmatrix = xgb.DMatrix(X)
    return model.predict(dmatrix)
