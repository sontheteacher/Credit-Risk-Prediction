import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def label_encoder(df, columns):
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df


