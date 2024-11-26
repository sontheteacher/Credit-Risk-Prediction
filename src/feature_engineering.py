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

def transform_categorical(accepted_data: pd.DataFrame) -> pd.DataFrame:
    """
    This function transforms categorical variables into numerical variables using one-hot encoding.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    transformed_data = accepted_data.copy()

    # Turn emp_length into numerical data
    transformed_data['emp_length'] = transformed_data['emp_length'].map(
        lambda x: 10.0 if x == "10+ years" else (0.5 if x == "< 1 year" else float(str(x).split()[0]))
    )

    # Turn term into numerical data
    transformed_data['term'] = transformed_data['term'].map(
        lambda x: float(str(x).split()[0])
    )

    # Turn earliest_cr_line into numerical data (using 2019-01-01 as benchmark date)
    transformed_data["earliest_cr_line"] = pd.to_datetime(transformed_data["earliest_cr_line"], format="%b-%Y")
    benchmark_date = pd.Timestamp("2019-01-01")
    transformed_data["earliest_cr_line"] = transformed_data["earliest_cr_line"].apply(
        lambda x: (np.timedelta64((x - benchmark_date), 'D').astype(int)) / -365
    )

    # One-hot encode other categorical variables
    one_hot_encoded_column = [
        "grade", "home_ownership", "verification_status", 
        "purpose", "initial_list_status", "application_type", "addr_state"
    ]

    transformed_data_numerical = pd.get_dummies(
        transformed_data, 
        columns=one_hot_encoded_column, 
        drop_first=True, 
        sparse=True
    )

    accepted_data["loan_status"] = accepted_data["loan_status"].apply(lambda x: True if x == "Fully Paid" or x=="Does not meet the credit policy. Status:Charged Off" else False)
    transformed_data_numerical["fully_paid"] = accepted_data["loan_status"]
    transformed_data_numerical = transformed_data_numerical.drop('loan_status', axis=1)

    return transformed_data_numerical


def compute_anomaly_score(accepted_data: pd.DataFrame, rejected_data : pd.DataFrame)->pd.DataFrame:
    """
    This function computes the anomaly score for each record in the dataset.
    """
    # Calculate the anomaly score
    
    accepted_data_subset = accepted_data[["loan_amnt", "dti", "addr_state_AL", "addr_state_AR", "addr_state_AZ", "addr_state_CA", "addr_state_CO", "addr_state_CT", "addr_state_DC", "addr_state_DE", "addr_state_FL", "addr_state_GA", "addr_state_HI", "addr_state_IA", "addr_state_ID", "addr_state_IL", "addr_state_IN", "addr_state_KS", "addr_state_KY", "addr_state_LA", "addr_state_MA", "addr_state_MD", "addr_state_ME", "addr_state_MI", "addr_state_MN", "addr_state_MO", "addr_state_MS", "addr_state_MT", "addr_state_NC", "addr_state_ND", "addr_state_NE", "addr_state_NH", "addr_state_NJ", "addr_state_NM", "addr_state_NV", "addr_state_NY", "addr_state_OH", "addr_state_OK", "addr_state_OR", "addr_state_PA", "addr_state_RI", "addr_state_SC", "addr_state_SD", "addr_state_TN", "addr_state_TX", "addr_state_UT", "addr_state_VA", "addr_state_VT", "addr_state_WA", "addr_state_WI", "addr_state_WV", "addr_state_WY", "emp_length"]] 

    accepted_data_subset.columns =  ['Amount Requested', 'Debt-To-Income Ratio', 'State_AL', 'State_AR',
                                    'State_AZ', 'State_CA', 'State_CO', 'State_CT', 'State_DC', 'State_DE',
                                    'State_FL', 'State_GA', 'State_HI', 'State_IA', 'State_ID', 'State_IL',
                                    'State_IN', 'State_KS', 'State_KY', 'State_LA', 'State_MA', 'State_MD',
                                    'State_ME', 'State_MI', 'State_MN', 'State_MO', 'State_MS', 'State_MT',
                                    'State_NC', 'State_ND', 'State_NE', 'State_NH', 'State_NJ', 'State_NM',
                                    'State_NV', 'State_NY', 'State_OH', 'State_OK', 'State_OR', 'State_PA',
                                    'State_RI', 'State_SC', 'State_SD', 'State_TN', 'State_TX', 'State_UT',
                                    'State_VA', 'State_VT', 'State_WA', 'State_WI', 'State_WV', 'State_WY', "Employment Length"]
    
    rejected_data_subset = rejected_data[['Amount Requested', 'Debt-To-Income Ratio', 'State_AL', 'State_AR',
                                    'State_AZ', 'State_CA', 'State_CO', 'State_CT', 'State_DC', 'State_DE',
                                    'State_FL', 'State_GA', 'State_HI', 'State_IA', 'State_ID', 'State_IL',
                                    'State_IN', 'State_KS', 'State_KY', 'State_LA', 'State_MA', 'State_MD',
                                    'State_ME', 'State_MI', 'State_MN', 'State_MO', 'State_MS', 'State_MT',
                                    'State_NC', 'State_ND', 'State_NE', 'State_NH', 'State_NJ', 'State_NM',
                                    'State_NV', 'State_NY', 'State_OH', 'State_OK', 'State_OR', 'State_PA',
                                    'State_RI', 'State_SC', 'State_SD', 'State_TN', 'State_TX', 'State_UT',
                                    'State_VA', 'State_VT', 'State_WA', 'State_WI', 'State_WV', 'State_WY', "Employment Length"]]

    rejected_data_subset = rejected_data_subset.copy()
    # Assuming employment length in rejected data is in string format
    rejected_data_subset["Employment Length"] = rejected_data_subset["Employment Length"].map(lambda x: 10.0 if x == "10+ years" else (1 if x == "< 1 year" else float(str(x).split()[0])))

    accepted_data_subset.loc[:, "Debt-To-Income Ratio"] = accepted_data_subset["Debt-To-Income Ratio"] / 100
    rejected_data_subset.loc[:, "Debt-To-Income Ratio"] = (
    rejected_data_subset["Debt-To-Income Ratio"].str.rstrip("%").astype("float") / 100
    )

    model = create_isolation_forest(accepted_data_subset)
    anomaly_score_accepted = model.decision_function(accepted_data_subset)
    
    batch_size = 1000000
    anomaly_scores = []  # List to store anomaly scores for all batches

    # Process the data in batches
    for i in range(0, len(rejected_data_subset), batch_size):
        batch = rejected_data_subset[i:i+batch_size]  # Extract batch
        anomaly_scores_batch = model.decision_function(batch)  # Compute scores for the batch
        anomaly_scores.extend(anomaly_scores_batch)  # Append scores to the list

    # Convert the list of scores into a single NumPy array or Pandas Series
    anomaly_score_rejected = np.array(anomaly_scores)  # Or use pd.Series(anomaly_score

    max_anomaly_score = max(max(anomaly_score_accepted), max(anomaly_score_rejected))
    normalized_anomaly_score_accepted = anomaly_score_accepted / max_anomaly_score

    accepted_data["anomaly_score"] = normalized_anomaly_score_accepted

    return accepted_data


def transform_rejected_categorical(data: pd.DataFrame) -> pd.DataFrame:
    """
    This function transforms categorical variables in the rejected data into numerical variables using one-hot encoding.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    transformed_data = data.copy()

    one_hot_encoded_column = ["State"]

    transformed_data_numerical = pd.get_dummies(
        transformed_data, 
        columns=one_hot_encoded_column, 
        drop_first=True, 
        sparse=True
    )

    return transformed_data_numerical