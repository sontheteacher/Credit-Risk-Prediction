from data_loader import load_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from grad_boost import grad_boost_model
from data_loader import load_data, load_data_as
from gpu_logistic import logistic_regression, weighted_logistic_regression
from visualize import calculate_best_models
import cupy as cp
import joblib

def test_grad_boost(X_train, X_test, y_train, y_test, path):
    # Test different configurations of the model
    n_estimators = [100, 500, 1000]
    max_depth = [3, 5, 7, 9]
    learning_rate = [0.01, 0.1, 0.05]
    results = []

    X_test = cp.array(X_test)

    best_accuracy = 0
    best_config = None

    for n in n_estimators:
        for d in max_depth:
            for l in learning_rate:
                model = grad_boost_model(X_train, y_train, n, d, l)
                y_pred = model.predict(X_test)
                y_pred = cp.asnumpy(y_pred)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"Configurations: n_estimators={n}, max_depth={d}, learning_rate={l}")
                print(f"Accuracy: {accuracy:.4f}")
                joblib.dump(model, path + f'XGboost_model_{n}_{d}_{l}.pkl')
                results.append((n, d, l, accuracy))
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_config = (n, d, l)
    print(f"XGB Best Accuracy: {best_accuracy:.4f}")
    print(f"Best Configuration: {best_config}")
    results.to_csv('models/XGB_model/results.csv')
    return

def test_logistic_regression(X_train, X_test, y_train, y_test , path):
    # Initialize and train the ML model
    model = logistic_regression(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Accuracy: {accuracy:.4f}")
    joblib.dump(model, path + 'logistic_regression_model.pkl')
    return

def test_weighted_logistic_regression(X_train, X_test, y_train, y_test, path):
    model = weighted_logistic_regression(X_train, y_train)

    y_pred = model.predict(X_test)

    # Step 5: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Weighted Logistic Regression Accuracy: {accuracy:.4f}")
    joblib.dump(model, path + 'weighted_logistic_regression_model.pkl')
    return

def train_model(flag):
    # Step 1: Import data
    file_path = 'data/finalized/accepted_data_finalized.csv'  # Provide the correct path to your data
    if flag == 1:
        xgb_path = 'models/XGB_model/as/'
        log_path = 'models/logistic_model/as/'
        X, y = load_data_as(file_path)
    else: 
        xgb_path = 'models/XGB_model/original/'
        log_path = 'models/logistic_model/original/'
        X, y = load_data(file_path)

    # Step 2: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train all models + print results
    test_logistic_regression(X_train, X_test, y_train, y_test, log_path)

    test_weighted_logistic_regression(X_train, X_test, y_train, y_test, log_path)

    test_grad_boost(X_train, X_test, y_train, y_test, xgb_path)

if __name__ == "__main__":
    # # Recover the same split
    X, y = load_data('data/finalized/accepted_data_finalized.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # logictic_model = joblib.load('models/logistic_model/original/logistic_regression_model.pkl')
    # wlogistic_model = joblib.load('models/logistic_model/original/weighted_logistic_regression_model.pkl')
    # lm_pred = logictic_model.predict(X_test)
    # wlm_pred = wlogistic_model.predict(X_test)
    # print(f"Logistic Model Accuracy: {accuracy_score(y_test, lm_pred)}")
    # print(f"Weighted Logistic Model Accuracy: {accuracy_score(y_test, wlm_pred)}")
    # # xgb_model_or, _ = get_best_model_roc_auc_from_folder('models/XGB_model/original/', X_test, y_test)
    # xgb_model_pr_rc, _ = get_best_model_precision_recall_auc_from_folder('models/XGB_model/original/', X_test, y_test)
    # log_model_pr_rc, _ = get_best_model_precision_recall_auc_from_folder('models/logistic_model/original/', X_test, y_test)
    # log_model_or, _ = get_best_model_precision_recall_auc_from_folder('models/logistic_model/as/', X_test, y_test)

    # Get the ROC and PR curves for the best models (out of the logistics)
    # roc_best_log, _ = calculate_best_models(0, 'models/logistic_model/original/', X_test, y_test, save_path="graphs/logistic_roc.png")
    # pr_best_log, _ = calculate_best_models(1, 'models/logistic_model/original/', X_test, y_test, save_path="graphs/logistic_pr.png")

    Xw, yw = load_data_as('data/finalized/accepted_data_finalized.csv')
    Xw_train, Xw_test, yw_train, yw_test = train_test_split(Xw, yw, test_size=0.2, random_state=42)
    # roc_best_wlog, _ = calculate_best_models(0, 'models/logistic_model/as/', Xw_test, yw_test, save_path="graphs/as_logistic_roc.png")
    # pr_best_wlog, _ = calculate_best_models(1, 'models/logistic_model/as/', Xw_test, yw_test, save_path="graphs/as_logistic_pr.png")

    xgb_best_roc, _ = calculate_best_models(0, 'models/XGB_model/original/', X_test, y_test, save_path="graphs/xgb_roc.png")
    xgb_best_pr, _ = calculate_best_models(1, 'models/XGB_model/original/', X_test, y_test, save_path="graphs/xgb_pr.png")
    xgb_as_roc, _ = calculate_best_models(0, 'models/XGB_model/as/', Xw_test, yw_test, save_path="graphs/as_xgb_roc.png")
    xgb_as_pr, _ = calculate_best_models(1, 'models/XGB_model/as/', Xw_test, yw_test, save_path="graphs/as_xgb_pr.png")


    