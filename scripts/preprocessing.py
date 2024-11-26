import sys
import os

# Add the project root directory to sys.path
project_root = os.path.abspath("..")  # Adjust this path if your notebook is in a different location
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
from sklearn.preprocessing import LabelEncoder

import src.data_preprocessing as prp
import src.feature_engineering as ftr

# Reload the module to reflect the changes
importlib.reload(prp)
importlib.reload(ftr)

if __name__ == "__main__":
    # Load the raw data
    print("Loading the raw data...")
    accepted_data = prp.load_accepted_data(finalized=False, processed=False)
    rejected_data = prp.load_rejected_data()

    print("Preprocessing the data...")
    # Drop features that would not be available at the time of loan application (cheat features)
    drop_list = ["issue_d", 'acc_now_delinq', 'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util', 'chargeoff_within_12_mths', 'collection_recovery_fee', 'collections_12_mths_ex_med', 'debt_settlement_flag', 'delinq_2yrs', 'delinq_amnt', 'disbursement_method', 'funded_amnt', 'funded_amnt_inv', 'hardship_flag', 'inq_last_6mths', 'last_credit_pull_d', 'last_fico_range_high', 'last_fico_range_low', 'last_pymnt_amnt', 'last_pymnt_d', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mths_since_recent_bc', 'mths_since_recent_inq', 'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m',  'out_prncp', 'out_prncp_inv', 'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'pymnt_plan', 'recoveries', 'tax_liens', 'tot_coll_amt', 'tot_cur_bal', 'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit', 'total_pymnt', 'total_pymnt_inv', 'total_rec_int', 'total_rec_late_fee', 'total_rec_prncp', 'total_rev_hi_lim', 'open_acc_6m', 'open_act_il', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m']
    cleaned_accepted_data = prp.drop_features(accepted_data, drop_list)

    cleaned_accepted_data = prp.drop_missing_features(cleaned_accepted_data, threshold=0.40)

    print("Feature engineering...")
    # Drop unrelated features and some other features to avoid high correlation
    drop_list = ["url", "id", "policy_code", "title", "emp_title", "installment", "fico_range_low", "open_acc", "pub_rec_bankruptcies", "sub_grade", "zip_code"]
    # Drop features that are highly correlated
    cleaned_accepted_data = prp.drop_features(cleaned_accepted_data, drop_list)
    
    print("shape of cleaned_accepted_data after dropping some columns: ", cleaned_accepted_data.shape)
    print("imputing missing values...")
    # Impute missing values
    accepted_data, rejected_data = prp.impute_in_chunks(cleaned_accepted_data, rejected_data)

    print("Encoding categorical variables...")
    accepted_data_numerical = ftr.transform_categorical(accepted_data)
    rejected_data_numerical = ftr.transform_rejected_categorical(rejected_data)

    print("Computing anomaly score...")
    accepted_data_numerical_with_anomaly_score = ftr.compute_anomaly_score(accepted_data_numerical, rejected_data_numerical)

    print("saving the data...")
    # Save the preprocessed data
    accepted_data_numerical_with_anomaly_score.to_csv('../data/finalized/accepted_data_finalized_with_anomaly_score.csv', index=False)

