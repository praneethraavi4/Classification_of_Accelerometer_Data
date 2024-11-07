import pandas as pd
from utils import read_yaml, scaler_pca, encode_labels, scaler_data, clean_data
from feature_extractors import (
    extract_pca_features,
    extract_frequency_domain_features,
    extract_time_domain_features,
)
from model import Model
from sklearn.model_selection import train_test_split
import joblib
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    logging.info("Starting the data processing")
    config = read_yaml(r"D:\repos\Classification_of_Accelerometer_Data\config.yaml")
    healthy_data = pd.read_hdf(config["healthy_data"])
    Anomaly_data = pd.read_hdf(config["Anomaly_data"])
    Anomaly_data_labels = pd.read_csv(config["Anomaly_ground_truth"])
    healthy_data["anomaly"] = 0.0
    Anomaly_data_with_labels = pd.concat(
        [Anomaly_data, Anomaly_data_labels["anomaly"]], axis=1
    )
    combined_data = pd.concat([healthy_data, Anomaly_data_with_labels], axis=0)
    X = combined_data.iloc[:, 0:61440]
    Y = combined_data.iloc[:, 61440]
    logging.info("Data Processed. Encoding the labels")
    Y = encode_labels(Y)
    logging.info("Labels encoded.")
    logging.info("Extracting features...")
    time_domain_features = extract_time_domain_features(X)
    logging.info("Time domain features extracted.")
    frequency_domain_features = extract_frequency_domain_features(X)
    logging.info("Frequency domain features extracted.")
    X, pca_scaler = scaler_pca(X)
    joblib.dump(pca_scaler, "scaler_pca.pkl")
    principal_components = extract_pca_features(X)
    logging.info("PCA components extracted.")
    features = pd.concat(
        [time_domain_features, frequency_domain_features, principal_components], axis=1
    )
    logging.info("Combining all features into a final data set.")
    final_data = pd.concat([features, pd.DataFrame(Y)], axis=1)
    logging.info("Cleaning the data...")
    final_data = clean_data(final_data)
    logging.info("Data cleaned.")
    X = final_data.iloc[:, 0:23]
    Y = final_data.iloc[:, 23]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, train_size=config["train_size"]
    )
    X_train_scaled, X_test_scaled, data_scaler = scaler_data(X_train, X_test)
    joblib.dump(data_scaler, "scaler_data.pkl")
    logging.info("Training the model...")
    model = Model()
    model.fit_model(X_train_scaled, Y_train)
    logging.info("Evaluating the model...")
    accuracy, auc = model.evaluate_model(X_test_scaled, Y_test)
    print("The test accuracy from the XGBoost model is : ", accuracy)
    print("The auc from the XGBoost model is : ", auc)
    model.save_model("model.pkl")


if __name__ == "__main__":
    main()
