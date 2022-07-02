import os
import pandas as pd


def read_data():
    df_path_list = os.listdir("./data")
    df_list = [pd.read_csv(f"./data/{path}") for path in df_path_list]
    return df_list
