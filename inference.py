import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from utils import read_yaml
from feature_extractors import (
    extract_pca_features,
    extract_frequency_domain_features,
    extract_time_domain_features,
)

config = read_yaml(r"D:\repos\Classification_of_Accelerometer_Data\config.yaml")
unseen_data = pd.read_hdf(config["unseen_data"])


def predict(df):
    pca_scaler = joblib.load("scaler_pca.pkl")
    data_scaler = joblib.load("scaler_data.pkl")
    model = joblib.load("model.pkl")
    time_domain_features = extract_time_domain_features(df)
    frequency_domain_features = extract_frequency_domain_features(df)
    scaled_data_pca = pca_scaler.transform(df)
    principal_components = extract_pca_features(scaled_data_pca)
    features = pd.concat(
        [time_domain_features, frequency_domain_features, principal_components], axis=1
    )
    scaled_features = data_scaler.transform(features)
    predictions = model.predict(scaled_features)
    for i in range(len(predictions)):
        if predictions[i] == 0:
            print("The accelerometer data is normal")
        else:
            print("The accelerometer data is abnormal")


if __name__ == "__main__":
    predict(unseen_data)
