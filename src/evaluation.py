from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(y_true, y_pred):
    """
    Evaluate the predictions using common error metrics.
    Returns a dictionary with evaluation scores.
    """
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred)
    }
