import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_absolute_error # Although not used in this file directly
# --- Add TensorBoard Import ---
from torch.utils.tensorboard import SummaryWriter
# --- End Add ---
import os # Needed for callback path check

# --- NEW: TensorBoard Callback for XGBoost ---
class TensorBoardCallback(xgb.callback.TrainingCallback):
    def __init__(self, writer: SummaryWriter, run_name: str):
        """
        Custom XGBoost callback to log evaluation metrics to TensorBoard.

        Args:
            writer: An initialized torch.utils.tensorboard.SummaryWriter.
            run_name: A unique name for this run (used for grouping scalars).
        """
        self.writer = writer
        self.run_name = run_name

    def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        """Log evaluation results after each boosting iteration."""
        if not evals_log:
            return False # Continue training

        # evals_log is like: {'eval': {'mae': [0.1, 0.09]}}
        # We want to log the latest value for each metric in each dataset
        for data, metric_dict in evals_log.items(): # data is 'eval' in our case
            for metric_name, values in metric_dict.items():
                if values: # Check if list is not empty
                    latest_value = values[-1]
                    # Create a tag like 'mae/eval' or 'rmse/train'
                    tag = f"{metric_name}/{data}"
                    # Log scalar: tag, value, step
                    self.writer.add_scalar(tag, latest_value, epoch)
        return False # Return False to continue training
# --- End NEW Callback ---


def create_xgb_model(params=None):
    """
    Create and return default parameters for an XGBoost model.
    (Using the reverted parameters that worked well)
    """
    if params is None:
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "eta": 0.1,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42 # Seed will be overwritten in pipeline loop
        }
    return params

# --- MODIFIED: train_xgb_model to accept callbacks ---
def train_xgb_model(params, X_train, y_train, X_val, y_val, num_rounds=100, callbacks=None):
    """
    Trains an XGBoost model using the provided parameters.
    Early stopping is applied with a window of 10 rounds.
    Accepts custom callbacks (e.g., for TensorBoard).

    Args:
        params: Parameter dictionary for XGBoost.
        X_train, y_train: Training feature matrix and target vector. Can be numpy or DataFrame.
        X_val, y_val: Validation feature matrix and target vector. Can be numpy or DataFrame.
        num_rounds: Maximum number of boosting rounds.
        callbacks: A list of XGBoost callback functions (optional).

    Returns:
        model: The trained XGBoost model.
    """
    # Create DMatrices required by xgb.train
    feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else None
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    eval_list = [(dval, "eval")] # Evaluate on validation set

    print(f"Training XGBoost with max {num_rounds} rounds and early stopping...")
    model = xgb.train(params,
                      dtrain,
                      num_boost_round=num_rounds,
                      evals=eval_list,
                      early_stopping_rounds=10,
                      verbose_eval=False, # Set to False to avoid duplicate console output if callback logs
                      callbacks=callbacks) # Pass callbacks here

    print(f"XGBoost training stopped after {model.best_iteration + 1} rounds.")
    return model
# --- End MODIFIED ---

def predict_xgb_model(model, X):
    """
    Make predictions using a trained XGBoost model.
    """
    feature_names = list(X.columns) if hasattr(X, 'columns') else None
    dmatrix = xgb.DMatrix(X, feature_names=feature_names)
    best_iteration = getattr(model, 'best_iteration', 0) + 1
    return model.predict(dmatrix, iteration_range=(0, best_iteration))

