import cupy as cp
from cuml.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# We can keep numpy for cuML

def logistic_regression(train_data, train_labels):
    # Create an instance of the logistic regression model
    model = LogisticRegression(penalty='l2', max_iter = 1000)
    # Fit the model to the data
    model.fit(train_data, train_labels)
    return model


def weighted_logistic_regression(train_data, train_labels):
    # Create an instance of the weighted logistic regression model with default weights
    model = LogisticRegression(penalty='l2', class_weight='balanced', max_iter=1000)
    # Fit the model to the data
    model.fit(train_data, train_labels)
    return model
