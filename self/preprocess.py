import os
import numpy as np
import pandas as pd
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, RobustScaler
from sklearn.model_selection import train_test_split

from config import *


# Fill in missing data
def handle_missing(df):


    # Drop the missing values in a column
    df = df.dropna(subset=DROP_NA_COL_LIST)

    # Fill with the mode
    for col in FILL_COL_LIST:
        a = df[col].rolling(3).mean()
        b = df.iloc[::-1][col].rolling(3).mean()
        c = a.fillna(b).fillna(df[col]).interpolate(method="nearest").ffill().bfill()
        df[col] = df[col].fillna(c)
    return df

# Extract feature
def extract_feature(df):
    # Drop the No col
    df = df.drop("No", axis=1)

    # Get whether is rain or not
    df["IS_RAIN"] = df.apply(lambda row: 0 if row["RAIN"] < 2.5 else 1, axis=1)

    # Get the date
    df["date"] = df.apply(lambda x: pd.to_datetime(f"{x['year']}-{x['month']}-{x['day']}"), axis=1)

    # Get the week
    df["week"] = df.apply(lambda x: x["date"].week, axis=1)

    # Get the day of week
    df["week_day"] = df.apply(lambda x: x["date"].dayofweek, axis=1)

    # Drop the date col
    df = df.drop("date", axis=1)

    return df


def split_df(df_list):
    # Reset the index
    df_list = [df.reset_index(drop=True) for df in df_list]

    # Concat all the dataframe
    df = pd.concat(df_list)

    # Sort the X by date
    df = df.sort_values(["year", "month", "day"])

    # Get X and Y
    X = df.drop("PM2.5", axis=1)
    y = df["PM2.5"]

    train_size = round(df.shape[0] * TRAIN_SIZE)

    X_train = X[:train_size]
    X_val = X[train_size:]
    y_train = y[:train_size]
    y_val = y[train_size:]

    return X_train, X_val, y_train, y_val


def transform_data(X_train, X_val, y_train, y_val):

    X_transformer = ColumnTransformer([
        ("log_transform", FunctionTransformer(np.log1p), LOG_TRANSFORM_COL_LIST),
        ("one_hot_encoder", OneHotEncoder(drop="if_binary", sparse=False, handle_unknown="ignore"),
         ONEHOT_ENCODER_COL_LIST),
        ("standard_scaler_transformer", StandardScaler(), STANDARD_SCALER_COL_LIST),
        ("robust_scaler_transformer", RobustScaler(), ROBUST_SCALER_COL_LIST)
    ])

    X_train = X_transformer.fit_transform(X_train)
    X_val = X_transformer.transform(X_val)

    y_train = np.log1p(y_train).to_numpy()
    y_val = np.log1p(y_val).to_numpy()

    return X_train, X_val, y_train, y_val


def preprocess_data(df_list):
    # Handle missing values
    df_list = [handle_missing(df) for df in df_list]

    # Extract feature
    df_list = [extract_feature(df) for df in df_list]


    # Split df
    X_train, X_val, y_train, y_val = split_df(df_list)

    # Transformer
    X_train, X_val, y_train, y_val = transform_data(X_train, X_val, y_train, y_val)

    return X_train, X_val, y_train, y_val


# if __name__ == "__main__":
#     # Read all data
#     df_path_list = os.listdir("./data")
#     df_list = [pd.read_csv(f"./data/{path}") for path in df_path_list]
#
#     X_train, X_val, y_train, y_val = preprocess(df_list)
#
#     print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
#     print(type(X_train), type(X_val), type(y_train), type(y_val))
