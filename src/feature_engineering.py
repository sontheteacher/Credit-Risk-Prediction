import pandas as pd
import numpy as np
import os
from sklearn.ensemble import IsolationForest

def anomaly_score(data : pd.DataFrame)->pd.DataFrame:
    """
    This function calculates the anomaly score for each record in the dataset.
    """
    # Calculate the anomaly score
    clf = IsolationForest()
    return clf.fit_predict(data)








