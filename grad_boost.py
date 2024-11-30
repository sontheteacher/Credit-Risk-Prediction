import xgboost as xgb
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cupy as cp

def grad_boost_model(train_data, train_labels, n_estimators, max_depth, learning_rate):
    X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.1, random_state=42)
    X_train, y_train = cp.array(train_data), cp.array(train_labels)
    X_val, y_val = cp.array(X_val), cp.array(y_val)
    # Create the model
    xgb_model = xgb.XGBClassifier(
        device = 'cuda',
        n_estimators = n_estimators, # big dataset
        max_depth = max_depth,
        learning_rate = learning_rate,
        objective = 'binary:logistic',
        subsample = 0.7,
        early_stopping_rounds = 10,
    )
    # Fit the model
    xgb_model.fit(X_train, y_train, eval_set = [(X_val, y_val)], verbose=False)
    return xgb_model
