import pandas as pd
import numpy as np

# Function to load data
def load_data(file_path):
    data = pd.read_csv(file_path)  # Replace with actual data loading logic
    X = data.drop('fully_paid', axis=1)  # Features
    y = data['fully_paid']               # Target
    X = X.drop(columns=['anomaly_score'])
    return X.to_numpy(), y.to_numpy()

def load_data_as(file_path):
    data = pd.read_csv(file_path)  # Replace with actual data loading logic
    X = data.drop('fully_paid', axis=1)  # Features
    y = data['fully_paid']               # Target
    return X.to_numpy(), y.to_numpy()
    

    
