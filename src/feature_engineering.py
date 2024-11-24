import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

def anomaly_score(model, data : pd.DataFrame)->pd.DataFrame:
    """
    This function calculates the anomaly score for each record in the dataset.
    """
    # Define the isolation forest
    
    return model.predict(data)

def create_isolation_forest(data: pd.DataFrame)->IsolationForest:
    """
    This function creates the isolation forest model.
    """
    # Define the isolation forest
    model = IsolationForest(n_estimators=100, max_samples='auto', contamination='auto', max_features=1.0, random_state=42)
    
    # Fit the model
    model.fit(data)
    
    return model

def scale_data(data: pd.DataFrame)->pd.DataFrame:
    """
    This function scales the data using MinMaxScaler.
    """
    # Define the scaler
    scaler = MinMaxScaler()
    
    # Fit and transform the data
    scaled_data = scaler.fit_transform(data)
    
    return pd.DataFrame(scaled_data, columns=data.columns)










