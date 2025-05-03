import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_absolute_error # Although not used in this file directly

def create_xgb_model(params=None):
    """
    Create and return default parameters for an XGBoost model optimized for time series forecasting.
    Reverted to original working parameters.
    """
    if params is None:
        # --- Reverted to Original Parameters ---
        params = {
            "objective": "reg:squarederror",  # using squared error objective
            "eval_metric": "mae",             # track MAE during training
            "eta": 0.1,                       # Original learning rate
            "max_depth": 70,                  # Original tree depth
            "subsample": 0.8,                 # Original subsample
            "colsample_bytree": 0.8,          # Original colsample
            "seed": 42                        # Keep seed for reproducibility
        }
        # --- End Revert ---
    # If params are passed (e.g., from Optuna), use those instead
    return params

def train_xgb_model(params, X_train, y_train, X_val, y_val, num_rounds=100):
    """
    Trains an XGBoost model using the provided parameters.
    Early stopping is applied with a window of 10 rounds.

    Args:
        params: Parameter dictionary for XGBoost.
        X_train, y_train: Training feature matrix and target vector. Can be numpy or DataFrame.
        X_val, y_val: Validation feature matrix and target vector. Can be numpy or DataFrame.
        num_rounds: Maximum number of boosting rounds.

    Returns:
        model: The trained XGBoost model.
    """
    # Create DMatrices required by xgb.train
    # Enable feature names if X_train/X_val are pandas DataFrames
    feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else None
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    eval_list = [(dval, "eval")] # Evaluate on validation set

    print(f"Training XGBoost with max {num_rounds} rounds and early stopping...")
    model = xgb.train(params,
                      dtrain,
                      num_boost_round=num_rounds,
                      evals=eval_list,
                      early_stopping_rounds=10, # Reverted early stopping rounds
                      verbose_eval=True) # Keep verbose to see progress

    print(f"XGBoost training stopped after {model.best_iteration + 1} rounds.")
    return model

def predict_xgb_model(model, X):
    """
    Make predictions using a trained XGBoost model.
    """
    # Enable feature names if X is a pandas DataFrame and model was trained with names
    feature_names = list(X.columns) if hasattr(X, 'columns') else None
    dmatrix = xgb.DMatrix(X, feature_names=feature_names)
    # Use best_ntree_limit if available (from early stopping)
    best_iteration = getattr(model, 'best_iteration', 0) + 1 # Use best iteration + 1
    return model.predict(dmatrix, iteration_range=(0, best_iteration))

