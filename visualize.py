import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from joblib import dump, load

def calculate_roc_auc(y_true, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    return auc(fpr, tpr)

def calculate_pr_auc(y_true, y_pred_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    return auc(recall, precision)

def calculate_best_models(flag, folder_path, X_test, y_test, save_path="best_model_plot.png"):
    """
    Identifies and saves the ROC or Precision-Recall curve for the best model (based on AUC).

    Parameters:
    - flag: int, 0 for ROC, 1 for PR
    - folder_path: str, path to the folder containing the saved models (.pkl).
    - X_test: numpy array or pandas DataFrame, test features.
    - y_test: numpy array or pandas Series, true labels.
    - save_path: str, path to save the graph image.

    Returns:
    - best_model: The model with the highest AUC score.
    - best_model_name: The filename of the best model.
    """
    best_auc = 0
    best_model = None
    best_model_name = ""

    # Ensure binary labels
    if not set(y_test).issubset({0, 1}):
        raise ValueError("y_test must contain only binary labels (0 and 1).")

    # Invert labels to treat minor class as positive
    y_test = 1 - y_test

    # Validate folder path
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder path '{folder_path}' does not exist.")

    skipped_models = []

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".pkl"):
            model_path = os.path.join(folder_path, filename)
            
            # Load the model
            with open(model_path, 'rb') as file:
                model = load(file)

            try:
                y_pred_prob = model.predict_proba(X_test)
                y_pred_prob_negative = y_pred_prob[:, 0]  # Probability for class 0, since we want to treat the minor class as positive
            except AttributeError:
                skipped_models.append(filename)
                continue

            # Compute the curve and AUC
            if flag == 0:
                fpr, tpr, _ = roc_curve(y_test, y_pred_prob_negative)
                roc_auc = auc(fpr, tpr)
            else:
                precision, recall, _ = precision_recall_curve(y_test, y_pred_prob_negative)
                roc_auc = auc(recall, precision)

            # Update the best model
            if roc_auc - best_auc > 0.01:
                best_auc = roc_auc
                best_model = model
                best_model_name = filename

    if best_model is not None:
        # Save the curve of the best model
        plt.figure(figsize=(10, 8))
        if flag == 0:
            plt.plot(fpr, tpr, color='b', lw=2, label=f'Best Model: {best_model_name} (AUC = {best_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve - Best Model')
        else:
            plt.plot(recall, precision, color='b', lw=2, label=f'Best Model: {best_model_name} (AUC = {best_auc:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve - Best Model')

        plt.legend(loc='best')
        plt.grid()

        # Save the plot to the specified path
        plt.savefig(save_path, format='png', dpi=300)
        plt.close()  # Close the plot to free memory
        print(f"Graph saved to {save_path}")
    else:
        print("No valid models found in the folder.")

    print(f"Skipped Models: {skipped_models}")
    return best_model, best_model_name