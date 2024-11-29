import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import tqdm
import joblib


def load_accepted_data(finalized=True, processed=True):
    if finalized:
        if processed:
            return pd.read_csv('../../data/finalized/accepted_data_finalized_with_anomaly_score.csv')
        else:
            return pd.read_csv('../../data/finalized/accepted_data_finalized.csv')
    else:
        return pd.read_csv('../../data/raw/accepted_data.csv')

def split_data(data):
    X = data.drop('fully_paid', axis=1)
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

    return {"accuracy": accuracy, "f1": f1, "roc_auc": roc_auc, "precision": precision, "recall": recall }

def plot_feature_importance(model, X_train):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = X_train.columns
    for i in range(X_train.shape[1]):
        print(f"{features[i]}: {importances[indices[i]]}")

import matplotlib.pyplot as plt
plt.show(block=True)
from tqdm import tqdm

def finetune_model(model, X_train, y_train, X_test, y_test, param_grid):
    best_score = 0
    best_params = None
    scores = []

    # Create a list of parameter combinations
    from itertools import product
    param_combinations = [
        dict(zip(params.keys(), values)) 
        for params in param_grid 
        for values in product(*params.values())
    ]

    # Iterate over parameter combinations with a progress bar
    for params in tqdm(param_combinations, desc="Fine-tuning Progress"):
        model.set_params(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        scores.append(score)

        if score > best_score:
            best_score = score
            best_params = params

    # Plotting the progress
    plt.plot(range(len(scores)), scores)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy Score')
    plt.title('Model Finetuning Progress')
    plt.show()

    print(f"Best Score: {best_score}")
    print(f"Best Parameters: {best_params}")

    return model.set_params(**best_params)

if __name__ == "__main__":
    data = load_accepted_data()

    X_train, X_test, y_train, y_test = split_data(data)

    model = define_model()
    model = train_model(model, X_train, y_train)

    print("Model evaluation...")
    print(" ")
    
    accuracy_train = evaluate_model(model, X_train, y_train)
    accuracy_test = evaluate_model(model, X_test, y_test)

    print(f"Accuracy on train data: {accuracy_train["accuracy"]}")
    print(f"f1 on train data: {accuracy_train["f1"]}")
    print(f"roc_auc on train data: {accuracy_train["roc_auc"]}")
    print(f"precision on train data: {accuracy_train["precision"]}")
    print(f"recall on train data: {accuracy_train["recall"]}")

    print(" ")

    print(f"Accuracy on test data: {accuracy_test["accuracy"]}")
    print(f"f1 on test data: {accuracy_test["f1"]}")
    print(f"roc_auc on test data: {accuracy_test["roc_auc"]}")
    print(f"precision on test data: {accuracy_test["precision"]}")
    print(f"recall on test data: {accuracy_test["recall"]}")

    # Save the model to a file
    model_filename = 'random_forest_model_1.pkl'
    joblib.dump(model, model_filename)
    print(f"Model saved to {model_filename}")

    print("Model finetuning complete.")

