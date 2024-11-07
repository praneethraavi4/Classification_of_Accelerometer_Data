from scipy.stats import skew, kurtosis
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import acf
import warnings


def calculate_rms(series):
    return np.sqrt(np.mean(series**2))


def zero_crossing_rate(series):
    return ((series[:-1] * series[1:]) < 0).sum()


def extract_time_domain_features(df, max_lag=5):
    features = pd.DataFrame()
    warnings.filterwarnings(
        "ignore", category=RuntimeWarning, message="invalid value encountered in divide"
    )

    for i in range(len(df)):
        series = df.iloc[i, 0:61440]

        # Basic statistical features (unchanged)
        features.loc[i, "mean"] = np.mean(series)
        features.loc[i, "std_dev"] = np.std(series)
        features.loc[i, "variance"] = np.var(series)
        features.loc[i, "min"] = np.min(series)
        features.loc[i, "max"] = np.max(series)
        features.loc[i, "skewness"] = skew(series) if np.var(series) > 1e-6 else np.nan
        features.loc[i, "kurtosis"] = (
            kurtosis(series) if np.var(series) > 1e-6 else np.nan
        )
        features.loc[i, "rms"] = calculate_rms(series)
        features.loc[i, "zero_crossing_rate"] = zero_crossing_rate(series)

        # Extract ACF features with proper check for avf[0] == 0
        acf_values = acf(series, nlags=max_lag)

        # Ensure no division by zero for acf normalization
        if np.abs(acf_values[0]) < 1e-6:
            acf_values = np.nan_to_num(
                acf_values
            )  # Replace zero or very small values with NaN or zero

        for lag in range(1, max_lag + 1):
            features.loc[i, f"acf_lag_{lag}"] = acf_values[lag]

    return features


def extract_frequency_domain_features(df, sampling_rate=1024):
    features = pd.DataFrame()
    n = 61440

    for i in range(len(df)):
        series = df.iloc[i, 0:61440]
        fft_values = np.fft.fft(series)
        fft_magnitude = np.abs(fft_values[: n // 2])
        freqs = np.fft.fftfreq(n, d=1 / sampling_rate)[: n // 2]

        # Avoid log(0) by ensuring that the magnitude is not zero
        fft_magnitude_safe = (
            fft_magnitude + 1e-12
        )  # Add a small constant to avoid zero magnitude

        # Extract frequency-domain features
        features.loc[i, "dominant_freq"] = freqs[np.argmax(fft_magnitude)]
        features.loc[i, "total_power"] = np.sum(fft_magnitude**2)

        # Spectral Entropy Calculation with safe log
        features.loc[i, "spectral_entropy"] = -np.sum(
            (fft_magnitude_safe / np.sum(fft_magnitude_safe))
            * np.log(fft_magnitude_safe / np.sum(fft_magnitude_safe))
        )

        features.loc[i, "freq_variance"] = np.var(fft_magnitude)

    return features


def extract_pca_features(df):
    pca = PCA(n_components=5)
    principalComponents = pca.fit_transform(df)
    principalDf = pd.DataFrame(
        data=principalComponents,
        columns=[
            "principal_component_1",
            "principal _component_2",
            "principal _component_3",
            "principal_component_4",
            "principal _component_5",
        ],
    )
    return principalDf
