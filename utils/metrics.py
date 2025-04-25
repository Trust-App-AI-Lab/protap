import evaluate

def eval_mse(y_true, y_pred):
    """Evaluate mse/rmse and return the results.
    squared: bool, default=True
        If True returns MSE value, if False returns RMSE value.
    """
    mse_metric = evaluate.load("mse")
    
    return mse_metric.compute(predictions=y_pred, references=y_true)

def eval_pearson(y_true, y_pred):
    """Evaluate Pearson correlation and return the results."""
    pearson_metric = evaluate.load("pearsonr")
    
    return pearson_metric.compute(predictions=y_pred, references=y_true)