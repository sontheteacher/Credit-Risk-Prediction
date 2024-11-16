import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt

def evaluate_classification_model(model, X_test, y_test):
    """
    Evaluate the classification model using various metrics.
    Args:
        model: The trained model to evaluate.
        X_test: Test features.
        y_test: True labels for the test set.
    Returns:
        dict: A dictionary containing the evaluation metrics.
    """
    # Predict the target on the test set
    y_pred = model.predict(X_test)
    
    # Calculate the evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])  # For binary classification
    
    # Return the metrics in a dictionary
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }

def plot_roc_curve(model, X_test, y_test):
    """
    Plot the ROC curve for the model.
    Args:
        model: The trained model to evaluate.
        X_test: Test features.
        y_test: True labels for the test set.
    """
    y_prob = model.predict_proba(X_test)[:, 1]  # Get predicted probabilities for the positive class
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label='ROC curve')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

def cross_val_evaluation(model, X, y, cv=5):
    """
    Perform k-fold cross-validation to evaluate the model.
    Args:
        model: The model to evaluate.
        X: Features for cross-validation.
        y: Target variable for cross-validation.
        cv: The number of folds in cross-validation.
    Returns:
        list: Cross-validation scores.
    """
    # Use stratified k-fold cross-validation for balanced class distribution
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv_strategy, scoring='accuracy')
    
    return cv_scores

def get_prediction_accuracy(model, X_test, y_test):
    """
    Get the accuracy of the model's predictions.
    Args:
        model: The trained model to evaluate.
        X_test: Test features.
        y_test: True labels for the test set.
    Returns:
        float: The accuracy of the model's predictions.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


