import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, roc_curve, precision_recall_curve, auc
import tqdm
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from itertools import product

def load_accepted_data(finalized=True, processed=True):
    if finalized:
        if processed:
            return pd.read_csv('../../data/finalized/accepted_data_finalized_with_anomaly_score.csv')
        else:
            return pd.read_csv('../../data/finalized/accepted_data_finalized.csv')
    else:
        return pd.read_csv('../../data/raw/accepted_data.csv')

def split_data(data, include_anomaly_score=True):
    X = data.drop('fully_paid', axis=1)
    if not include_anomaly_score:
        X = X.drop('anomaly_score', axis=1)

    y = data['fully_paid']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def define_model():
    # return RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, min_samples_split=2)
    return RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        max_depth=25, 
        min_samples_split=4, 
        min_samples_leaf=2, 
        max_features='sqrt'
    )

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='blue')
    plt.plot([0, 1], [0, 1], 'k--', label='Random chance')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')

    # Save the figure
    plt.savefig('roc_curve.png', dpi=300)  # Save as PNG with 300 DPI
    plt.show()  # Display the plot

    return {"accuracy": accuracy, "f1": f1, "roc_auc": roc_auc, "precision": precision, "recall": recall }

def plot_feature_importance(model, X_train):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = X_train.columns
    for i in range(X_train.shape[1]):
        print(f"{features[i]}: {importances[indices[i]]}")

def train_and_evaluate_hyperparameters(hyperparameters, X_train, y_train, X_test, y_test):
    results = []
    for params in hyperparameters:
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        metrics_train = {
            "accuracy": accuracy_score(y_train, y_pred_train),
            "f1": f1_score(y_train, y_pred_train),
            "roc_auc": roc_auc_score(y_train, y_pred_train),
            "precision": precision_score(y_train, y_pred_train),
            "recall": recall_score(y_train, y_pred_train)
        }
        
        metrics_test = {
            "accuracy": accuracy_score(y_test, y_pred_test),
            "f1": f1_score(y_test, y_pred_test),
            "roc_auc": roc_auc_score(y_test, y_pred_test),
            "precision": precision_score(y_test, y_pred_test),
            "recall": recall_score(y_test, y_pred_test)
        }
        
        results.append({
            "params": params,
            "metrics_train": metrics_train,
            "metrics_test": metrics_test
        })
        
        model_filename = f'random_forest_model_{params["n_estimators"]}_{params["max_depth"]}_{params["min_samples_split"]}_{params["min_samples_leaf"]}_{params["max_features"]}.pkl'
        joblib.dump(model, model_filename)
        print(f"Model with params {params} saved to {model_filename}")
        print(f"Train metrics: {metrics_train}")
        print(f"Test metrics: {metrics_test}")
        print(" ")
    
    return results

def evaluate_wrt_label0():
    data = load_accepted_data()

    X_train, X_test, y_train, y_test = split_data(data)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    n_estimators = [150, 250]
    max_depth = [10, 20, 30]
    min_samples_split = [2, 4, 6]
    min_samples_leaf = [2, 3]

    hyperparameters = [
        {"n_estimators": n, "max_depth": d, "min_samples_split": s, "min_samples_leaf": l, "max_features": "sqrt"}
        for n, d, s, l in product(n_estimators, max_depth, min_samples_split, min_samples_leaf)
    ]
    with open('random_forest_evaluate_class0.txt', 'w') as f:
        for params in hyperparameters:
            model_filename = f'random_forest_model_{params["n_estimators"]}_{params["max_depth"]}_{params["min_samples_split"]}_{params["min_samples_leaf"]}_{params["max_features"]}.pkl'
            model = joblib.load(model_filename)
            
            y_pred_train_class1 = model.predict(X_train)
            y_pred_test_class1 = model.predict(X_val)

            y_pred_train_class0 = 1 - y_pred_train_class1
            y_pred_test_class0 = 1 - y_pred_test_class1

            roc_auc_for_class0_train = roc_auc_score(y_train, y_pred_train_class0)
            roc_auc_for_class0_test = roc_auc_score(y_val, y_pred_test_class0)

            # precision-recall curve for class 0
            precision_train_class0, recall_train_class0, threshold_train_class0 = precision_recall_curve(y_train, y_pred_train_class0, pos_label=0)
            precision_train_class1, recall_train_class1, threshold_train_class1 = precision_recall_curve(y_val, y_pred_test_class0, pos_label=0)

            # precision-recall auc for class 0
            precision_recall_curve_for_class0_train = auc(recall_train_class0, precision_train_class0)
            precision_recall_curve_for_class0_test = auc(recall_train_class1, precision_train_class1)

            f.write(f"Model: {model_filename}\n")
            f.write(f"ROC AUC for class 0 on train data: {roc_auc_for_class0_train}\n")
            f.write(f"ROC AUC for class 0 on validation data: {roc_auc_for_class0_test}\n")
            f.write(f"Precision-Recall AUC for class 0 on train data: {precision_recall_curve_for_class0_train}\n")
            f.write(f"Precision-Recall AUC for class 0 on validation data: {precision_recall_curve_for_class0_test}\n")
            f.write("\n")


def main2():
    data = load_accepted_data()

    X_train, X_test, y_train, y_test = split_data(data)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    n_estimators = [150, 250]
    max_depth = [10, 20, 30]
    min_samples_split = [2, 4, 6]
    min_samples_leaf = [2, 3]

    hyperparameters = [
        {"n_estimators": n, "max_depth": d, "min_samples_split": s, "min_samples_leaf": l, "max_features": "sqrt"}
        for n, d, s, l in product(n_estimators, max_depth, min_samples_split, min_samples_leaf)
    ]

    results = train_and_evaluate_hyperparameters(hyperparameters, X_train, y_train, X_val, y_val)
    with open('hyperparameter_results.txt', 'w') as f:
        for result in results:
            f.write(f"Params: {result['params']}\n")
            f.write(f"Train metrics: {result['metrics_train']}\n")
            f.write(f"Validation metrics: {result['metrics_test']}\n")
            f.write("\n")



def main():
    data = load_accepted_data()

    X_train, X_test, y_train, y_test = split_data(data)

    model = define_model()
    model = train_model(model, X_train, y_train)

    print("Model evaluation...")
    print(" ")
    
    accuracy_train = evaluate_model(model, X_train, y_train)
    accuracy_test = evaluate_model(model, X_test, y_test)

    print(f'Accuracy on train data: {accuracy_train["accuracy"]}')
    print(f'f1 on train data: {accuracy_train["f1"]}')
    print(f'roc_auc on train data: {accuracy_train["roc_auc"]}')
    print(f'precision on train data: {accuracy_train["precision"]}')
    print(f'recall on train data: {accuracy_train["recall"]}')

    print(" ")

    print(f'Accuracy on test data: {accuracy_test["accuracy"]}')
    print(f'f1 on test data: {accuracy_test["f1"]}')
    print(f'roc_auc on test data: {accuracy_test["roc_auc"]}')
    print(f'precision on test data: {accuracy_test["precision"]}')
    print(f'recall on test data: {accuracy_test["recall"]}')

    # Save the model to a file
    model_filename = 'random_forest_model_1.pkl'
    joblib.dump(model, model_filename)
    print(f"Model saved to {model_filename}")

    print("Model finetuning complete.")

if __name__ == "__main__":
    # main()
    # main2()
    evaluate_wrt_label0()

