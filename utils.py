import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from typing import Tuple


def read_yaml(path_to_yaml: str) -> dict:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
        return content


def scaler_pca(X: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    sc = StandardScaler()
    X = sc.fit_transform(X)
    return X, sc


def scaler_data(
    X: pd.DataFrame, Y: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    sc = MinMaxScaler()
    X = sc.fit_transform(X)
    Y = sc.transform(Y)
    return X, Y, sc


def encode_labels(X: pd.DataFrame) -> pd.DataFrame:
    le = LabelEncoder()
    X = le.fit_transform(X)
    return X


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df.fillna(0)
    df.drop_duplicates(inplace=True)
    return df
