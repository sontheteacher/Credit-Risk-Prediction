import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def logistic_regression(train_data, train_labels):
    # Create an instance of the logistic regression model
    model = LogisticRegression(penalty='l2', solver='sag', random_state=42, n_jobs=-1)
    # Fit the model to the data
    model.fit(train_data, train_labels)
    return model

def weighted_logistic_regression(train_data, train_labels):
    # Create an instance of the weighted logistic regression model with default weights
    model = LogisticRegression(penalty='l2', solver='sag', class_weight='default', random_state=69, n_jobs=-1)
    # Fit the model to the data
    model.fit(train_data, train_labels)
    return model