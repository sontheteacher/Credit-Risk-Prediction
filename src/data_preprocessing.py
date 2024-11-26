import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

def load_data(finalized = False, processed = False)->pd.DataFrame:

    rejected_data = pd.read_csv('..\\data\\raw\\rejected\\rejected_2007_to_2018Q4.csv')

    if finalized:
        accepted_data = pd.read_csv('..\\data\\finalized\\final_accepted.csv')
        return accepted_data, rejected_data

    if processed:
        accepted_data = pd.read_csv('..\\data\\processed\\preprocessed_accepted.csv')
    else:
        accepted_data = pd.read_csv('..\\data\\raw\\accepted\\accepted_2007_to_2018Q4.csv')
        
    return accepted_data, rejected_data

def load_accepted_data(finalized = False, processed = False)->pd.DataFrame:
    processed_chunks = []
    
    if finalized:
        chunks = pd.read_csv('..\\data\\finalized\\final_accepted.csv')
    if processed:
        chunks = pd.read_csv('..\\data\\processed\\preprocessed_accepted.csv')
    else:
        chunks = pd.read_csv('..\\data\\raw\\accepted\\accepted_2007_to_2018Q4.csv')
    
    chunk_size = 100000
        
    for start in range(0, chunks.shape[0], chunk_size):
        end = start + chunk_size
        chunk = chunks[start:end].copy()
        # Process each chunk (e.g., filter out 'Late (31-120 days)' rows)
        chunk = chunk[chunk['loan_status'] != 'Late (31-120 days)']
        chunk = chunk[chunk['loan_status'] != 'Late (16-30 days)']
        chunk = chunk[chunk['loan_status'] != 'In Grace Period']
        chunk = chunk[chunk['loan_status'] != 'Current']

        processed_chunks.append(chunk)

    return pd.concat(processed_chunks, ignore_index=True)


def load_rejected_data()->pd.DataFrame:

    chunk_size = 100000  # Adjust based on your system's memory
    chunks = []

    for chunk in pd.read_csv('..\\data\\raw\\rejected\\rejected_2007_to_2018Q4.csv', chunksize=chunk_size, low_memory=False):
        # Perform processing on each chunk if needed
        chunks.append(chunk)

    rejected_data = pd.concat(chunks, ignore_index=True)
    return rejected_data

def drop_missing_features(data: pd.DataFrame, threshold = 0.5)->pd.DataFrame:

    """Drop features missing too much data based on the threshold"""
    # Total number of missing values
    missing_val = data.isnull().sum()

    # Percentage of missing values
    missing_val_percent = missing_val / len(data)

    # Type of missing values
    missing_val_type = data.dtypes

    mis_val_table = pd.concat([missing_val, missing_val_percent, missing_val_type], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0: 'Missing Values', 1: 'Percent of Total Values', 2: 'Data Type'})

    drop_list = sorted(mis_val_table_ren_columns[mis_val_table_ren_columns["Percent of Total Values"] > threshold].index)

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[ mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('Percent of Total Values', ascending=False).round(1)
        
    # Print some summary information
    print ("Your selected dataframe has " + str(data.shape[1]) + " columns.\n" "There are " + str(mis_val_table_ren_columns.shape[0]) + " columns that have missing values.")
    print ("There are " + str(len(drop_list)) + " columns to be dropped.")

    # Return the dataframe with missing information
    return data.drop(drop_list, axis=1)


def drop_features(data: pd.DataFrame, drop_list)->pd.DataFrame:

    """Drop features that would not be available at the time of loan application"""

    # drop_list = ['acc_now_delinq', 'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util', 'chargeoff_within_12_mths', 'collection_recovery_fee', 'collections_12_mths_ex_med', 'debt_settlement_flag', 'delinq_2yrs', 'delinq_amnt', 'disbursement_method', 'funded_amnt', 'funded_amnt_inv', 'hardship_flag', 'inq_last_6mths', 'last_credit_pull_d', 'last_fico_range_high', 'last_fico_range_low', 'last_pymnt_amnt', 'last_pymnt_d', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mths_since_recent_bc', 'mths_since_recent_inq', 'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m',  'out_prncp', 'out_prncp_inv', 'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'pymnt_plan', 'recoveries', 'tax_liens', 'tot_coll_amt', 'tot_cur_bal', 'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit', 'total_pymnt', 'total_pymnt_inv', 'total_rec_int', 'total_rec_late_fee', 'total_rec_prncp', 'total_rev_hi_lim', 'open_acc_6m', 'open_act_il', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m']
    # drop_list = ['acc_now_delinq', 'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util', 'chargeoff_within_12_mths', 'collection_recovery_fee', 'collections_12_mths_ex_med', 'debt_settlement_flag', 'delinq_2yrs', 'delinq_amnt', 'disbursement_method', 'funded_amnt', 'funded_amnt_inv', 'hardship_flag', 'inq_last_6mths', 'last_credit_pull_d', 'last_fico_range_high', 'last_fico_range_low', 'last_pymnt_amnt', 'last_pymnt_d', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mths_since_recent_bc', 'mths_since_recent_inq', 'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m',  'out_prncp', 'out_prncp_inv', 'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'pymnt_plan', 'recoveries', 'tax_liens', 'tot_coll_amt', 'tot_cur_bal', 'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit', 'total_pymnt', 'total_pymnt_inv', 'total_rec_int', 'total_rec_late_fee', 'total_rec_prncp', 'total_rev_hi_lim']

    cleaned_data = data.drop(drop_list, axis=1)

    return cleaned_data


def split_data(data: pd.DataFrame)->pd.DataFrame:

    """Split the dataset into training and testing sets."""

    X = data.drop('loan_status', axis=1)
    y = data['loan_status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def plot_corr_matrix(data: pd.DataFrame)->None:

    plt.figure(figsize=(12, 8))
    numeric_data = data.select_dtypes(include=[np.number])
    sns.heatmap(numeric_data.corr(), annot=True, cmap='viridis')
    plt.show()


def remove_percent_sign(input: str)->float:
    data = float(input.rstrip('%')) / 100.0
    return data

def impute_data(accepted_data, rejected_data)->None:

    # Impute missing values in accepted_data
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    # Separate numerical and categorical columns
    numerical_cols = accepted_data.select_dtypes(include=['float64']).columns
    categorical_cols = accepted_data.select_dtypes(include=['object']).columns

    accepted_data[numerical_cols] = num_imputer.fit_transform(accepted_data[numerical_cols])
    accepted_data[categorical_cols] = cat_imputer.fit_transform(accepted_data[categorical_cols])

    # Impute missing values in rejected_data
    numerical_cols_rejected = rejected_data.select_dtypes(include=['float64']).columns
    categorical_cols_rejected = rejected_data.select_dtypes(include=['object']).columns

    rejected_data[numerical_cols_rejected] = num_imputer.fit_transform(rejected_data[numerical_cols_rejected])
    rejected_data[categorical_cols_rejected] = cat_imputer.fit_transform(rejected_data[categorical_cols_rejected])

    return accepted_data, rejected_data

def impute_in_chunks(accepted_data, rejected_data, chunk_size=100000):

    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    # Separate numerical and categorical columns
    numerical_cols = accepted_data.select_dtypes(include=['float64']).columns
    categorical_cols = accepted_data.select_dtypes(include=['object']).columns

    chunks_1 = []
    for start in range(0, accepted_data.shape[0], chunk_size):
        end = start + chunk_size
        chunk = accepted_data[start:end].copy()

        chunk[numerical_cols] = num_imputer.fit_transform(chunk[numerical_cols])
        chunk[categorical_cols] = cat_imputer.fit_transform(chunk[categorical_cols])

        # imputed_chunk = pd.DataFrame(imputer.fit_transform(chunk), columns=data.columns)
        chunks_1.append(chunk)
    
    # Impute missing values in rejected_data
    numerical_cols_rejected = rejected_data.select_dtypes(include=['float64']).columns
    categorical_cols_rejected = rejected_data.select_dtypes(include=['object']).columns

    chunks_2 = []
    for start in range(0, rejected_data.shape[0], chunk_size):
        end = start + chunk_size
        chunk = rejected_data[start:end].copy()

        chunk[numerical_cols_rejected] = num_imputer.fit_transform(chunk[numerical_cols_rejected])
        chunk[categorical_cols_rejected] = cat_imputer.fit_transform(chunk[categorical_cols_rejected])

        # imputed_chunk = pd.DataFrame(imputer.fit_transform(chunk), columns=data.columns)
        chunks_2.append(chunk)

    accepted_data = pd.concat(chunks_1, ignore_index=True)
    rejected_data = pd.concat(chunks_2, ignore_index=True)

    return accepted_data, rejected_data


def drop_late(data: pd.DataFrame)->pd.DataFrame:

    "Drop loans that are on going, late, or in grace period"

    data = data[data['loan_status'] != 'Late (16-30 days)']
    data = data[data['loan_status'] != 'Late (31-120 days)']
    data = data[data['loan_status'] != 'In Grace Period']
    data = data[data['loan_status'] != 'Current']

    return data