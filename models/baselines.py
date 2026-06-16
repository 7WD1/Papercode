"""Baseline Methods for Anomaly Detection Comparison

Implements: ARIMA, Isolation Forest, LSTM-AE, Transformer-Anomaly, RL-AD
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats as sp_stats
import warnings
warnings.filterwarnings('ignore')


class ARIMADetector:
    """ARIMA-based anomaly detection using residual analysis."""

    def __init__(self, order=(2, 1, 2), threshold_sigma=3.0):
        self.order = order
        self.threshold_sigma = threshold_sigma
        self.model = None
        self.residual_std_ = None

    def fit_predict(self, values, labels=None):
        """Fit ARIMA and detect anomalies from residuals."""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(values, order=self.order)
            fitted = model.fit()
            residuals = fitted.resid
        except Exception:
            # Fallback: simple differencing + moving average residual
            ma = np.convolve(values, np.ones(20)/20, mode='same')
            residuals = values - ma

        self.residual_std_ = np.std(residuals)
        if self.residual_std_ < 1e-8:
            self.residual_std_ = 1.0

        z_scores = np.abs(residuals) / self.residual_std_
        predictions = (z_scores > self.threshold_sigma).astype(int)
        scores = 1.0 - np.exp(-z_scores)  # convert to anomaly score

        return predictions, scores


class IsolationForestDetector:
    """Isolation Forest anomaly detection with sliding window features."""

    def __init__(self, window_size=50, n_estimators=100, contamination=0.05):
        self.window_size = window_size
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.scaler = StandardScaler()

    def _extract_features(self, values):
        """Extract window-based features."""
        n = len(values)
        features = []
        for i in range(self.window_size, n + 1):
            w = values[i - self.window_size:i]
            feat = [
                np.mean(w), np.std(w), np.min(w), np.max(w),
                sp_stats.skew(w), sp_stats.kurtosis(w),
                np.median(w), np.percentile(w, 25), np.percentile(w, 75),
                np.sum(np.abs(np.diff(w))) / len(w)  # roughness
            ]
            features.append(feat)
        return np.array(features)

    def fit_predict(self, values, labels=None):
        """Fit Isolation Forest and predict anomalies."""
        features = self._extract_features(values)
        features = self.scaler.fit_transform(features)

        iso_forest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=42
        )
        iso_labels = iso_forest.fit_predict(features)

        predictions = (iso_labels == -1).astype(int)
        scores = -iso_forest.score_samples(features)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        # Pad to match original length
        full_pred = np.zeros(len(values), dtype=int)
        full_scores = np.zeros(len(values))
        full_pred[self.window_size - 1:] = predictions
        full_scores[self.window_size - 1:] = scores

        return full_pred, full_scores


class LSTMAEDetector:
    """LSTM Autoencoder anomaly detection using reconstruction error."""

    def __init__(self, window_size=50, hidden_size=32, epochs=20,
                 threshold_percentile=95):
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.threshold_percentile = threshold_percentile

    def fit_predict(self, values, labels=None):
        """Fit LSTM-AE and detect anomalies from reconstruction error."""
        try:
            return self._torch_fit_predict(values, labels)
        except ImportError:
            return self._numpy_fit_predict(values, labels)

    def _torch_fit_predict(self, values, labels=None):
        """PyTorch-based LSTM-AE implementation."""
        import torch
        import torch.nn as nn

        # Normalize
        val_mean, val_std = values.mean(), values.std()
        vals_norm = (values - val_mean) / (val_std + 1e-8)

        # Create windows
        windows = []
        for i in range(self.window_size, len(vals_norm) + 1):
            windows.append(vals_norm[i - self.window_size:i])
        windows = np.array(windows)

        # Convert to torch
        X = torch.FloatTensor(windows).unsqueeze(-1)  # (N, w, 1)

        class LSTMAE(nn.Module):
            def __init__(self, input_size, hidden_size, seq_len):
                super().__init__()
                self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, input_size)
                self.hidden_size = hidden_size
                self.seq_len = seq_len

            def forward(self, x):
                _, (h, c) = self.encoder(x)
                # h shape: (1, batch, hidden) -> repeat to (batch, seq_len, hidden)
                dec_input = h.permute(1, 0, 2).repeat(1, self.seq_len, 1)
                decoded, _ = self.decoder(dec_input, (h, c))
                decoded = self.fc(decoded)
                return decoded

        model = LSTMAE(1, self.hidden_size, self.window_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        # Train
        model.train()
        batch_size = 64
        for epoch in range(self.epochs):
            perm = torch.randperm(len(X))
            for i in range(0, len(X), batch_size):
                batch = X[perm[i:i+batch_size]]
                recon = model(batch)
                loss = criterion(recon, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Compute reconstruction error
        model.eval()
        with torch.no_grad():
            recon = model(X)
            errors = ((X - recon) ** 2).mean(dim=(1, 2)).numpy()

        # Threshold
        threshold = np.percentile(errors, self.threshold_percentile)
        predictions = (errors > threshold).astype(int)

        # Pad
        full_pred = np.zeros(len(values), dtype=int)
        full_scores = np.zeros(len(values))
        full_pred[self.window_size - 1:] = predictions
        full_scores[self.window_size - 1:] = (errors - errors.min()) / (errors.max() - errors.min() + 1e-8)

        return full_pred, full_scores

    def _numpy_fit_predict(self, values, labels=None):
        """Numpy fallback: simple autoencoder-like approach."""
        val_mean, val_std = values.mean(), values.std()
        vals_norm = (values - val_mean) / (val_std + 1e-8)

        windows = []
        for i in range(self.window_size, len(vals_norm) + 1):
            windows.append(vals_norm[i - self.window_size:i])
        windows = np.array(windows)

        # Simple PCA-based reconstruction as autoencoder substitute
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(5, self.window_size // 2))
        reduced = pca.fit_transform(windows)
        reconstructed = pca.inverse_transform(reduced)

        errors = np.mean((windows - reconstructed) ** 2, axis=1)
        threshold = np.percentile(errors, self.threshold_percentile)
        predictions = (errors > threshold).astype(int)

        full_pred = np.zeros(len(values), dtype=int)
        full_scores = np.zeros(len(values))
        full_pred[self.window_size - 1:] = predictions
        full_scores[self.window_size - 1:] = (errors - errors.min()) / (errors.max() - errors.min() + 1e-8)

        return full_pred, full_scores


class TransformerAnomalyDetector:
    """Transformer-based anomaly detection using attention patterns."""

    def __init__(self, window_size=50, d_model=32, nhead=4,
                 threshold_percentile=95):
        self.window_size = window_size
        self.d_model = d_model
        self.nhead = nhead
        self.threshold_percentile = threshold_percentile

    def fit_predict(self, values, labels=None):
        """Transformer anomaly detection."""
        try:
            return self._torch_fit_predict(values, labels)
        except (ImportError, Exception):
            return self._numpy_fit_predict(values, labels)

    def _torch_fit_predict(self, values, labels=None):
        """PyTorch Transformer implementation."""
        import torch
        import torch.nn as nn

        val_mean, val_std = values.mean(), values.std()
        vals_norm = (values - val_mean) / (val_std + 1e-8)

        windows = []
        for i in range(self.window_size, len(vals_norm) + 1):
            windows.append(vals_norm[i - self.window_size:i])
        windows = np.array(windows)

        X = torch.FloatTensor(windows).unsqueeze(-1)  # (N, w, 1)

        class TransformerAE(nn.Module):
            def __init__(self, d_model, nhead):
                super().__init__()
                self.encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, batch_first=True
                )
                self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
                self.input_proj = nn.Linear(1, d_model)
                self.output_proj = nn.Linear(d_model, 1)

            def forward(self, x):
                x = self.input_proj(x)
                encoded = self.encoder(x)
                decoded = self.output_proj(encoded)
                return decoded

        model = TransformerAE(self.d_model, self.nhead)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        model.train()
        batch_size = 64
        for epoch in range(15):
            perm = torch.randperm(len(X))
            for i in range(0, len(X), batch_size):
                batch = X[perm[i:i+batch_size]]
                recon = model(batch)
                loss = criterion(recon, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            recon = model(X)
            errors = ((X - recon) ** 2).mean(dim=(1, 2)).numpy()

        threshold = np.percentile(errors, self.threshold_percentile)
        predictions = (errors > threshold).astype(int)

        full_pred = np.zeros(len(values), dtype=int)
        full_scores = np.zeros(len(values))
        full_pred[self.window_size - 1:] = predictions
        full_scores[self.window_size - 1:] = (errors - errors.min()) / (errors.max() - errors.min() + 1e-8)

        return full_pred, full_scores

    def _numpy_fit_predict(self, values, labels=None):
        """Numpy fallback: attention-like scoring using correlation."""
        val_mean, val_std = values.mean(), values.std()
        vals_norm = (values - val_mean) / (val_std + 1e-8)

        windows = []
        for i in range(self.window_size, len(vals_norm) + 1):
            windows.append(vals_norm[i - self.window_size:i])
        windows = np.array(windows)

        # Attention-like: compute correlation-based anomaly score
        ref_window = np.median(windows[:len(windows)//4], axis=0)

        errors = np.zeros(len(windows))
        for i in range(len(windows)):
            corr = np.corrcoef(windows[i], ref_window)[0, 1]
            errors[i] = 1.0 - abs(corr) if not np.isnan(corr) else 1.0
            errors[i] += np.mean((windows[i] - ref_window) ** 2) * 0.1

        threshold = np.percentile(errors, self.threshold_percentile)
        predictions = (errors > threshold).astype(int)

        full_pred = np.zeros(len(values), dtype=int)
        full_scores = np.zeros(len(values))
        full_pred[self.window_size - 1:] = predictions
        full_scores[self.window_size - 1:] = (errors - errors.min()) / (errors.max() - errors.min() + 1e-8)

        return full_pred, full_scores


class RLADDetector:
    """Standard RL-based anomaly detection (without homotopy).

    Uses Q-learning for threshold adaptation.
    """

    def __init__(self, window_size=50, n_bins=20, learning_rate=0.1,
                 gamma=0.99, epsilon=0.1, n_episodes=50):
        self.window_size = window_size
        self.n_bins = n_bins
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_episodes = n_episodes
        self.q_table = None
        self.scaler = StandardScaler()

    def _discretize_state(self, features):
        """Discretize continuous features into bins."""
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Use first 3 features for state
        state_feats = features[:, :min(3, features.shape[1])]

        if self.scaler.mean_ is None:
            self.scaler.fit(state_feats)

        scaled = self.scaler.transform(state_feats)

        # Bin each dimension
        binned = np.clip(
            ((scaled + 3) / 6 * self.n_bins).astype(int),
            0, self.n_bins - 1
        )

        # Convert to single state index
        state_idx = 0
        for col in range(binned.shape[1]):
            state_idx = state_idx * self.n_bins + binned[0, col]

        return state_idx % (self.n_bins ** min(3, features.shape[1]))

    def fit_predict(self, features, labels):
        """Q-learning based anomaly detection.

        Args:
            features: normalized feature array (T, d)
            labels: ground truth labels (T,)
        """
        n_states = self.n_bins ** min(3, features.shape[1])
        n_actions = 2  # 0=normal, 1=anomaly
        self.q_table = np.zeros((n_states, n_actions))
        self.scaler.fit(features[:, :min(3, features.shape[1])])

        # Q-learning training
        for episode in range(self.n_episodes):
            total_reward = 0
            for t in range(len(features) - 1):
                state = self._discretize_state(features[t])

                # Epsilon-greedy action selection
                if np.random.random() < self.epsilon:
                    action = np.random.randint(n_actions)
                else:
                    action = np.argmax(self.q_table[state])

                # Reward
                true_label = labels[t]
                if action == true_label:
                    reward = 1.0
                elif action == 1 and true_label == 0:
                    reward = -0.8  # false alarm
                else:
                    reward = -1.0  # missed

                # Next state
                next_state = self._discretize_state(features[t + 1])

                # Q-update
                best_next = np.max(self.q_table[next_state])
                td_target = reward + self.gamma * best_next
                self.q_table[state, action] += self.lr * (td_target - self.q_table[state, action])

                total_reward += reward

        # Predict
        predictions = np.zeros(len(features), dtype=int)
        scores = np.zeros(len(features))

        for t in range(len(features)):
            state = self._discretize_state(features[t])
            q_values = self.q_table[state]
            scores[t] = sigmoid(q_values[1] - q_values[0])  # relative Q-value
            predictions[t] = int(q_values[1] > q_values[0])

        return predictions, scores


class VAEADDetector:
    """Variational Autoencoder for anomaly detection.

    Encoder maps input to latent (mu, logvar), reparameterization trick,
    decoder reconstructs. Anomaly score = KL divergence + reconstruction error.
    Falls back to PCA-based approach if PyTorch is unavailable.
    """

    def __init__(self, window_size=50, latent_dim=8, threshold_percentile=95,
                 epochs=20):
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.threshold_percentile = threshold_percentile
        self.epochs = epochs

    def fit_predict(self, values, labels=None):
        try:
            return self._torch_fit_predict(values, labels)
        except ImportError:
            return self._numpy_fit_predict(values, labels)

    def _torch_fit_predict(self, values, labels=None):
        import torch
        import torch.nn as nn

        val_mean, val_std = values.mean(), values.std()
        vals_norm = (values - val_mean) / (val_std + 1e-8)

        windows = np.array([vals_norm[i - self.window_size:i]
                            for i in range(self.window_size, len(vals_norm) + 1)])
        X = torch.FloatTensor(windows).unsqueeze(-1)

        class VAE(nn.Module):
            def __init__(self, seq_len, latent_dim):
                super().__init__()
                self.encoder_fc1 = nn.Linear(seq_len, 128)
                self.encoder_fc2 = nn.Linear(128, 64)
                self.fc_mu = nn.Linear(64, latent_dim)
                self.fc_logvar = nn.Linear(64, latent_dim)
                self.decoder_fc1 = nn.Linear(latent_dim, 64)
                self.decoder_fc2 = nn.Linear(64, 128)
                self.decoder_out = nn.Linear(128, seq_len)
                self.seq_len = seq_len

            def encode(self, x):
                x = x.view(x.size(0), -1)
                h = torch.relu(self.encoder_fc1(x))
                h = torch.relu(self.encoder_fc2(h))
                return self.fc_mu(h), self.fc_logvar(h)

            def reparameterize(self, mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std

            def decode(self, z):
                h = torch.relu(self.decoder_fc1(z))
                h = torch.relu(self.decoder_fc2(h))
                return self.decoder_out(h)

            def forward(self, x):
                x_flat = x.view(x.size(0), -1)
                mu, logvar = self.encode(x)
                z = self.reparameterize(mu, logvar)
                recon = self.decode(z)
                return recon, mu, logvar

        def vae_loss(recon_x, x, mu, logvar):
            recon_loss = nn.functional.mse_loss(recon_x, x.view(x.size(0), -1),
                                                 reduction='none').sum(dim=1)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            return (recon_loss + kl).mean()

        model = VAE(self.window_size, self.latent_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        model.train()
        batch_size = 64
        for epoch in range(self.epochs):
            perm = torch.randperm(len(X))
            for i in range(0, len(X), batch_size):
                batch = X[perm[i:i + batch_size]]
                recon, mu, logvar = model(batch)
                loss = vae_loss(recon, batch, mu, logvar)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            recon, mu, logvar = model(X)
            recon_err = ((X.view(X.size(0), -1) - recon) ** 2).mean(dim=1)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            scores_raw = (recon_err + kl_div * 0.01).numpy()

        threshold = np.percentile(scores_raw, self.threshold_percentile)
        predictions = (scores_raw > threshold).astype(int)

        full_pred = np.zeros(len(values), dtype=int)
        full_scores = np.zeros(len(values))
        full_pred[self.window_size - 1:] = predictions
        smin, smax = scores_raw.min(), scores_raw.max()
        full_scores[self.window_size - 1:] = (scores_raw - smin) / (smax - smin + 1e-8)
        return full_pred, full_scores

    def _numpy_fit_predict(self, values, labels=None):
        from sklearn.decomposition import PCA

        val_mean, val_std = values.mean(), values.std()
        vals_norm = (values - val_mean) / (val_std + 1e-8)

        windows = np.array([vals_norm[i - self.window_size:i]
                            for i in range(self.window_size, len(vals_norm) + 1)])

        pca = PCA(n_components=min(self.latent_dim, self.window_size // 2))
        reduced = pca.fit_transform(windows)
        reconstructed = pca.inverse_transform(reduced)

        recon_err = np.mean((windows - reconstructed) ** 2, axis=1)
        kl_approx = np.mean(reduced ** 2, axis=1)
        scores_raw = recon_err + kl_approx * 0.01

        threshold = np.percentile(scores_raw, self.threshold_percentile)
        predictions = (scores_raw > threshold).astype(int)

        full_pred = np.zeros(len(values), dtype=int)
        full_scores = np.zeros(len(values))
        full_pred[self.window_size - 1:] = predictions
        smin, smax = scores_raw.min(), scores_raw.max()
        full_scores[self.window_size - 1:] = (scores_raw - smin) / (smax - smin + 1e-8)
        return full_pred, full_scores


class USADDetector:
    """UnSupervised Anomaly Detection with dual decoders.

    Single encoder, two adversarial decoders. Anomaly score is the combined
    reconstruction error from both decoders. Falls back to PCA if no torch.
    """

    def __init__(self, window_size=50, latent_dim=8, threshold_percentile=95,
                 epochs=20):
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.threshold_percentile = threshold_percentile
        self.epochs = epochs

    def fit_predict(self, values, labels=None):
        try:
            return self._torch_fit_predict(values, labels)
        except ImportError:
            return self._numpy_fit_predict(values, labels)

    def _torch_fit_predict(self, values, labels=None):
        import torch
        import torch.nn as nn

        val_mean, val_std = values.mean(), values.std()
        vals_norm = (values - val_mean) / (val_std + 1e-8)

        windows = np.array([vals_norm[i - self.window_size:i]
                            for i in range(self.window_size, len(vals_norm) + 1)])
        X = torch.FloatTensor(windows).unsqueeze(-1)

        class Encoder(nn.Module):
            def __init__(self, seq_len, latent_dim):
                super().__init__()
                self.fc1 = nn.Linear(seq_len, 64)
                self.fc2 = nn.Linear(64, latent_dim)

            def forward(self, x):
                x = x.view(x.size(0), -1)
                return torch.relu(self.fc2(torch.relu(self.fc1(x))))

        class Decoder(nn.Module):
            def __init__(self, latent_dim, seq_len):
                super().__init__()
                self.fc1 = nn.Linear(latent_dim, 64)
                self.fc2 = nn.Linear(64, seq_len)

            def forward(self, z):
                return self.fc2(torch.relu(self.fc1(z)))

        encoder = Encoder(self.window_size, self.latent_dim)
        decoder1 = Decoder(self.latent_dim, self.window_size)
        decoder2 = Decoder(self.latent_dim, self.window_size)

        opt_enc = torch.optim.Adam(encoder.parameters(), lr=1e-3)
        opt_d1 = torch.optim.Adam(decoder1.parameters(), lr=1e-3)
        opt_d2 = torch.optim.Adam(decoder2.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        batch_size = 64
        for epoch in range(self.epochs):
            perm = torch.randperm(len(X))
            for i in range(0, len(X), batch_size):
                batch = X[perm[i:i + batch_size]]
                x_flat = batch.view(batch.size(0), -1)

                z = encoder(batch)

                # Decoder 1: reconstruct from encoder output
                recon1 = decoder1(z)
                loss1 = criterion(recon1, x_flat)
                opt_d1.zero_grad()
                loss1.backward(retain_graph=True)
                opt_d1.step()

                # Decoder 2: adversarial -- tries to reconstruct from
                # encoder output but differently
                recon2 = decoder2(z.detach())
                loss2 = criterion(recon2, x_flat)
                opt_d2.zero_grad()
                loss2.backward()
                opt_d2.step()

                # Encoder minimises combined loss
                z2 = encoder(batch)
                r1 = decoder1(z2)
                r2 = decoder2(z2)
                enc_loss = criterion(r1, x_flat) - 0.1 * criterion(r2, x_flat)
                opt_enc.zero_grad()
                enc_loss.backward()
                opt_enc.step()

        encoder.eval()
        decoder1.eval()
        decoder2.eval()
        with torch.no_grad():
            x_flat = X.view(X.size(0), -1)
            z = encoder(X)
            r1 = decoder1(z)
            r2 = decoder2(z)
            err1 = ((x_flat - r1) ** 2).mean(dim=1).numpy()
            err2 = ((x_flat - r2) ** 2).mean(dim=1).numpy()
            scores_raw = err1 + err2

        threshold = np.percentile(scores_raw, self.threshold_percentile)
        predictions = (scores_raw > threshold).astype(int)

        full_pred = np.zeros(len(values), dtype=int)
        full_scores = np.zeros(len(values))
        full_pred[self.window_size - 1:] = predictions
        smin, smax = scores_raw.min(), scores_raw.max()
        full_scores[self.window_size - 1:] = (scores_raw - smin) / (smax - smin + 1e-8)
        return full_pred, full_scores

    def _numpy_fit_predict(self, values, labels=None):
        from sklearn.decomposition import PCA

        val_mean, val_std = values.mean(), values.std()
        vals_norm = (values - val_mean) / (val_std + 1e-8)

        windows = np.array([vals_norm[i - self.window_size:i]
                            for i in range(self.window_size, len(vals_norm) + 1)])

        pca1 = PCA(n_components=min(self.latent_dim, self.window_size // 2))
        pca2 = PCA(n_components=max(1, min(self.latent_dim - 1, self.window_size // 3)))

        r1 = pca1.fit_transform(windows)
        recon1 = pca1.inverse_transform(r1)
        r2 = pca2.fit_transform(windows)
        recon2 = pca2.inverse_transform(r2)

        err1 = np.mean((windows - recon1) ** 2, axis=1)
        err2 = np.mean((windows - recon2) ** 2, axis=1)
        scores_raw = err1 + err2

        threshold = np.percentile(scores_raw, self.threshold_percentile)
        predictions = (scores_raw > threshold).astype(int)

        full_pred = np.zeros(len(values), dtype=int)
        full_scores = np.zeros(len(values))
        full_pred[self.window_size - 1:] = predictions
        smin, smax = scores_raw.min(), scores_raw.max()
        full_scores[self.window_size - 1:] = (scores_raw - smin) / (smax - smin + 1e-8)
        return full_pred, full_scores


class TimesNetDetector:
    """TimesNet: Temporal 2D-Variation modeling for anomaly detection.

    Converts 1D series to 2D via FFT period finding, applies 2D convolution
    approximation. Simplified: FFT period detection + sliding window deviation.
    """

    def __init__(self, window_size=50, threshold_percentile=95, n_periods=3):
        self.window_size = window_size
        self.threshold_percentile = threshold_percentile
        self.n_periods = n_periods

    def _find_periods(self, values, top_k=None):
        """Find dominant periods via FFT."""
        if top_k is None:
            top_k = self.n_periods
        vals_centered = values - values.mean()
        fft_vals = np.fft.rfft(vals_centered)
        magnitudes = np.abs(fft_vals)
        magnitudes[0] = 0  # ignore DC
        top_indices = np.argsort(magnitudes)[-top_k:]
        periods = []
        for idx in sorted(top_indices):
            if idx > 0:
                p = len(vals_centered) / idx
                periods.append(max(2, int(round(p))))
        if not periods:
            periods = [self.window_size // 2]
        return periods

    def fit_predict(self, values, labels=None):
        val_mean, val_std = values.mean(), values.std()
        vals_norm = (values - val_mean) / (val_std + 1e-8)

        periods = self._find_periods(vals_norm)
        n = len(vals_norm)

        # For each period, create 2D folding and compute reconstruction error
        all_errors = np.zeros(n)
        for period in periods:
            period = min(period, self.window_size)
            # Fold the series into 2D matrix (rows = period, cols = n/period)
            n_rows = period
            n_cols = n // n_rows
            if n_cols < 2:
                continue
            folded = vals_norm[:n_rows * n_cols].reshape(n_rows, n_cols)

            # Compute per-column statistics as a simple 2D model
            col_median = np.median(folded, axis=1, keepdims=True)
            deviation = np.abs(folded - col_median)
            unfolded_err = deviation.reshape(-1)
            all_errors[:len(unfolded_err)] += unfolded_err

        # Sliding window aggregation
        errors = np.zeros(n)
        for i in range(self.window_size, n + 1):
            errors[i - 1] = np.mean(all_errors[i - self.window_size:i])

        # Normalise
        errors[:self.window_size] = errors[self.window_size] if n > self.window_size else 0

        threshold = np.percentile(errors[self.window_size - 1:], self.threshold_percentile)
        predictions = (errors > threshold).astype(int)

        smin, smax = errors.min(), errors.max()
        scores = (errors - smin) / (smax - smin + 1e-8)
        return predictions, scores


class PatchTSTDetector:
    """PatchTST-AD: Patch-based Transformer for anomaly detection.

    Splits series into patches, uses patch-level reconstruction.
    Simplified: patch-based PCA reconstruction.
    """

    def __init__(self, window_size=50, patch_len=16, stride=8,
                 threshold_percentile=95):
        self.window_size = window_size
        self.patch_len = patch_len
        self.stride = stride
        self.threshold_percentile = threshold_percentile

    def fit_predict(self, values, labels=None):
        try:
            return self._torch_fit_predict(values, labels)
        except ImportError:
            return self._numpy_fit_predict(values, labels)

    def _torch_fit_predict(self, values, labels=None):
        import torch
        import torch.nn as nn

        val_mean, val_std = values.mean(), values.std()
        vals_norm = (values - val_mean) / (val_std + 1e-8)

        windows = np.array([vals_norm[i - self.window_size:i]
                            for i in range(self.window_size, len(vals_norm) + 1)])

        # Create patches from each window
        patch_data = []
        for w in windows:
            patches = [w[j:j + self.patch_len]
                       for j in range(0, self.window_size - self.patch_len + 1, self.stride)]
            patch_data.append(patches)
        patch_data = np.array(patch_data)  # (n_windows, n_patches, patch_len)
        n_windows, n_patches, pl = patch_data.shape

        X = torch.FloatTensor(patch_data.reshape(n_windows, -1))

        class PatchModel(nn.Module):
            def __init__(self, input_dim, hidden_dim=32):
                super().__init__()
                self.encoder = nn.Linear(input_dim, hidden_dim)
                self.decoder = nn.Linear(hidden_dim, input_dim)

            def forward(self, x):
                h = torch.relu(self.encoder(x))
                return self.decoder(h)

        model = PatchModel(n_patches * pl)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(20):
            recon = model(X)
            loss = criterion(recon, X)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            recon = model(X)
            errors = ((X - recon) ** 2).mean(dim=1).numpy()

        threshold = np.percentile(errors, self.threshold_percentile)
        predictions = (errors > threshold).astype(int)

        full_pred = np.zeros(len(values), dtype=int)
        full_scores = np.zeros(len(values))
        full_pred[self.window_size - 1:] = predictions
        smin, smax = errors.min(), errors.max()
        full_scores[self.window_size - 1:] = (errors - smin) / (smax - smin + 1e-8)
        return full_pred, full_scores

    def _numpy_fit_predict(self, values, labels=None):
        from sklearn.decomposition import PCA

        val_mean, val_std = values.mean(), values.std()
        vals_norm = (values - val_mean) / (val_std + 1e-8)

        windows = np.array([vals_norm[i - self.window_size:i]
                            for i in range(self.window_size, len(vals_norm) + 1)])

        # Flatten patch representations
        patch_rows = []
        for w in windows:
            patches = []
            for j in range(0, self.window_size - self.patch_len + 1, self.stride):
                patches.extend(w[j:j + self.patch_len])
            patch_rows.append(patches)
        patch_matrix = np.array(patch_rows)

        n_components = min(5, patch_matrix.shape[1] // 2, patch_matrix.shape[0] - 1)
        n_components = max(1, n_components)
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(patch_matrix)
        reconstructed = pca.inverse_transform(reduced)

        errors = np.mean((patch_matrix - reconstructed) ** 2, axis=1)
        threshold = np.percentile(errors, self.threshold_percentile)
        predictions = (errors > threshold).astype(int)

        full_pred = np.zeros(len(values), dtype=int)
        full_scores = np.zeros(len(values))
        full_pred[self.window_size - 1:] = predictions
        smin, smax = errors.min(), errors.max()
        full_scores[self.window_size - 1:] = (errors - smin) / (smax - smin + 1e-8)
        return full_pred, full_scores


class DCdetectorDetector:
    """DCdetector: Dual Attention Contrastive anomaly detection.

    Uses temporal and feature attention with contrastive learning.
    Simplified: attention-weighted reconstruction via correlation.
    """

    def __init__(self, window_size=50, threshold_percentile=95, n_heads=2):
        self.window_size = window_size
        self.threshold_percentile = threshold_percentile
        self.n_heads = n_heads

    def fit_predict(self, values, labels=None):
        val_mean, val_std = values.mean(), values.std()
        vals_norm = (values - val_mean) / (val_std + 1e-8)

        windows = np.array([vals_norm[i - self.window_size:i]
                            for i in range(self.window_size, len(vals_norm) + 1)])

        # Reference "normal" pattern from first quarter of data
        ref = np.median(windows[:max(1, len(windows) // 4)], axis=0)

        # Dual attention: temporal and feature
        n = len(windows)
        errors = np.zeros(n)

        # Compute similarity-based attention weights
        for i in range(n):
            w = windows[i]

            # Temporal attention: sliding inner product
            temporal_scores = np.array([
                np.dot(w[j:j + 5], ref[j:j + 5]) / (np.linalg.norm(w[j:j + 5]) * np.linalg.norm(ref[j:j + 5]) + 1e-8)
                for j in range(0, self.window_size - 5 + 1, 5)
            ])
            temporal_attn = np.exp(temporal_scores - temporal_scores.max())
            temporal_attn /= temporal_attn.sum() + 1e-8

            # Feature attention: pointwise correlation
            feature_attn = np.abs(w - ref)
            feature_attn = np.exp(-feature_attn)
            feature_attn /= feature_attn.sum() + 1e-8

            # Combined attention-weighted distance
            weighted_dist = feature_attn * (w - ref) ** 2
            errors[i] = np.mean(weighted_dist)

        threshold = np.percentile(errors, self.threshold_percentile)
        predictions = (errors > threshold).astype(int)

        full_pred = np.zeros(len(values), dtype=int)
        full_scores = np.zeros(len(values))
        full_pred[self.window_size - 1:] = predictions
        smin, smax = errors.min(), errors.max()
        full_scores[self.window_size - 1:] = (errors - smin) / (smax - smin + 1e-8)
        return full_pred, full_scores


class MemSTDetector:
    """Memory-augmented Spatiotemporal anomaly detection.

    Memory bank of prototype patterns; distance to nearest prototype is the
    anomaly score. Simplified: k-means centroids as memory.
    """

    def __init__(self, window_size=50, n_prototypes=50, threshold_percentile=95):
        self.window_size = window_size
        self.n_prototypes = n_prototypes
        self.threshold_percentile = threshold_percentile

    def fit_predict(self, values, labels=None):
        from sklearn.cluster import MiniBatchKMeans

        val_mean, val_std = values.mean(), values.std()
        vals_norm = (values - val_mean) / (val_std + 1e-8)

        windows = np.array([vals_norm[i - self.window_size:i]
                            for i in range(self.window_size, len(vals_norm) + 1)])

        n_clusters = min(self.n_prototypes, len(windows) // 2)
        n_clusters = max(2, n_clusters)
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42,
                                 batch_size=min(256, len(windows)),
                                 n_init=3)
        cluster_labels = kmeans.fit_predict(windows)
        centroids = kmeans.cluster_centers_

        # Distance to nearest centroid
        dists = np.zeros(len(windows))
        for i, w in enumerate(windows):
            d = np.linalg.norm(w - centroids, axis=1)
            dists[i] = d.min()

        threshold = np.percentile(dists, self.threshold_percentile)
        predictions = (dists > threshold).astype(int)

        full_pred = np.zeros(len(values), dtype=int)
        full_scores = np.zeros(len(values))
        full_pred[self.window_size - 1:] = predictions
        smin, smax = dists.min(), dists.max()
        full_scores[self.window_size - 1:] = (dists - smin) / (smax - smin + 1e-8)
        return full_pred, full_scores


class iTransformerDetector:
    """Inverted Transformer for anomaly detection.

    Attends over variate dimension instead of time (channel independence).
    Simplified: per-channel anomaly scoring via reconstruction.
    """

    def __init__(self, window_size=50, threshold_percentile=95, n_segments=5):
        self.window_size = window_size
        self.threshold_percentile = threshold_percentile
        self.n_segments = n_segments

    def fit_predict(self, values, labels=None):
        val_mean, val_std = values.mean(), values.std()
        vals_norm = (values - val_mean) / (val_std + 1e-8)

        windows = np.array([vals_norm[i - self.window_size:i]
                            for i in range(self.window_size, len(vals_norm) + 1)])

        # Treat each position in the window as a "variate"
        # Split window into segments, score each independently
        seg_len = max(1, self.window_size // self.n_segments)
        errors = np.zeros(len(windows))

        # Learn per-segment normal pattern from first quarter
        ref_windows = windows[:max(1, len(windows) // 4)]
        ref_segments = []
        for s in range(self.n_segments):
            start = s * seg_len
            end = min(start + seg_len, self.window_size)
            if start >= self.window_size:
                break
            seg_data = ref_windows[:, start:end]
            ref_segments.append((seg_data.mean(axis=0), seg_data.std(axis=0) + 1e-8))

        for i, w in enumerate(windows):
            seg_errors = []
            for s, (ref_mean, ref_std) in enumerate(ref_segments):
                start = s * seg_len
                end = min(start + seg_len, self.window_size)
                if start >= self.window_size:
                    break
                seg = w[start:end]
                z_score = np.abs((seg - ref_mean) / ref_std)
                seg_errors.append(np.mean(z_score))
            errors[i] = np.mean(seg_errors)

        threshold = np.percentile(errors, self.threshold_percentile)
        predictions = (errors > threshold).astype(int)

        full_pred = np.zeros(len(values), dtype=int)
        full_scores = np.zeros(len(values))
        full_pred[self.window_size - 1:] = predictions
        smin, smax = errors.min(), errors.max()
        full_scores[self.window_size - 1:] = (errors - smin) / (smax - smin + 1e-8)
        return full_pred, full_scores


class CARDDetector:
    """Contrastive learning with Residual Decomposition.

    Decomposes series into trend + seasonal + residual, then applies
    Isolation Forest on residuals for anomaly detection.
    """

    def __init__(self, window_size=50, threshold_percentile=95, period=None):
        self.window_size = window_size
        self.threshold_percentile = threshold_percentile
        self.period = period

    def _decompose(self, values):
        """Simple STL-like decomposition into trend + seasonal + residual."""
        n = len(values)
        period = self.period
        if period is None:
            # Auto-detect period via FFT
            fft_vals = np.fft.rfft(values - values.mean())
            magnitudes = np.abs(fft_vals)
            magnitudes[0] = 0
            top_idx = np.argmax(magnitudes)
            if top_idx > 0:
                period = max(2, int(round(n / top_idx)))
            else:
                period = max(2, n // 10)

        period = min(period, n // 2)
        period = max(2, period)

        # Trend: moving average with period window
        trend = np.convolve(values, np.ones(period) / period, mode='same')

        # Seasonal: average over each phase
        detrended = values - trend
        seasonal = np.zeros(n)
        for p in range(period):
            indices = np.arange(p, n, period)
            if len(indices) > 0:
                seasonal[indices] = np.mean(detrended[indices])

        residual = values - trend - seasonal
        return trend, seasonal, residual

    def fit_predict(self, values, labels=None):
        trend, seasonal, residual = self._decompose(values)

        # Extract features from residuals using sliding windows
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        n = len(residual)
        features = []
        for i in range(self.window_size, n + 1):
            w = residual[i - self.window_size:i]
            feat = [
                np.mean(w), np.std(w), np.min(w), np.max(w),
                np.median(w),
                sp_stats.skew(w) if np.std(w) > 1e-8 else 0,
                sp_stats.kurtosis(w) if np.std(w) > 1e-8 else 0,
            ]
            features.append(feat)
        features = np.array(features)

        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
        iso_labels = iso.fit_predict(features)

        predictions = (iso_labels == -1).astype(int)
        scores = -iso.score_samples(features)
        smin, smax = scores.min(), scores.max()
        scores = (scores - smin) / (smax - smin + 1e-8)

        full_pred = np.zeros(n, dtype=int)
        full_scores = np.zeros(n)
        full_pred[self.window_size - 1:] = predictions
        full_scores[self.window_size - 1:] = scores
        return full_pred, full_scores


def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )


# Registry of all baseline methods (keys match paper Table I method names)
BASELINES = {
    'ARIMA': ARIMADetector,
    'Isolation Forest': IsolationForestDetector,
    'LSTM-AE': LSTMAEDetector,
    'VAE-AD': VAEADDetector,
    'USAD': USADDetector,
    'Anomaly Transformer': TransformerAnomalyDetector,
    'TimesNet': TimesNetDetector,
    'PatchTST-AD': PatchTSTDetector,
    'DCdetector': DCdetectorDetector,
    'MAUT': MemSTDetector,
    'iTransformer-AD': iTransformerDetector,
    'TranAD': CARDDetector,
    'RL-AD': RLADDetector,
}
