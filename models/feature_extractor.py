"""Feature Extraction Module for HRLAD
Implements multi-domain feature extraction with sliding window mechanism.
"""

import numpy as np
from scipy import stats as sp_stats


class FeatureExtractor:
    """Multi-domain feature extractor for time series anomaly detection.

    Extracts time-domain statistics and frequency-domain FFT features
    from sliding windows, with Z-score normalization.
    """

    def __init__(self, window_size=64, n_fft_components=4):
        self.window_size = window_size
        self.n_fft_components = n_fft_components
        self.feature_dim = 4 + n_fft_components  # 4 time-domain + m freq-domain (Paper Table III: d=8)
        self.mean_ = None
        self.std_ = None
        self.fitted = False

    def _extract_window_features(self, window):
        """Extract features from a single window of shape (window_size,)."""
        # Time-domain features
        mu = np.mean(window)
        var = np.var(window, ddof=1)
        skew = sp_stats.skew(window)
        kurt = sp_stats.kurtosis(window)

        # Frequency-domain features (FFT magnitude)
        fft_vals = np.fft.rfft(window)
        fft_mag = np.abs(fft_vals[1:self.n_fft_components + 1])  # skip DC

        # Pad if window is too short
        if len(fft_mag) < self.n_fft_components:
            fft_mag = np.pad(fft_mag, (0, self.n_fft_components - len(fft_mag)))

        features = np.concatenate([[mu, var, skew, kurt], fft_mag])
        return features

    def fit(self, time_series):
        """Fit the normalizer on training data.

        Args:
            time_series: 1D numpy array of shape (T,)
        Returns:
            self
        """
        features_list = []
        for i in range(self.window_size, len(time_series)):
            window = time_series[i - self.window_size:i]
            features_list.append(self._extract_window_features(window))

        features_array = np.array(features_list)
        self.mean_ = np.mean(features_array, axis=0)
        self.std_ = np.std(features_array, axis=0)
        self.std_[self.std_ < 1e-8] = 1e-8  # avoid division by zero
        self.fitted = True
        return self

    def transform(self, time_series):
        """Transform time series to feature sequences with sliding window.

        Args:
            time_series: 1D numpy array of shape (T,)
        Returns:
            features: normalized feature array of shape (T - window_size + 1, feature_dim)
            indices: corresponding time indices
        """
        if not self.fitted:
            raise RuntimeError("Must call fit() before transform()")

        features_list = []
        for i in range(self.window_size, len(time_series) + 1):
            window = time_series[i - self.window_size:i]
            feat = self._extract_window_features(window)
            features_list.append(feat)

        features_array = np.array(features_list)
        # Z-score normalization
        normalized = (features_array - self.mean_) / self.std_
        indices = np.arange(self.window_size - 1, len(time_series))
        return normalized, indices

    def fit_transform(self, time_series):
        """Fit and transform in one step."""
        self.fit(time_series)
        return self.transform(time_series)

    def extract_single(self, window):
        """Extract and normalize features from a single window."""
        if not self.fitted:
            raise RuntimeError("Must call fit() first")
        feat = self._extract_window_features(window)
        return (feat - self.mean_) / self.std_
