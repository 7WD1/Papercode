"""Markov Jump System Module for HRLAD
Implements transition probability estimation with EM algorithm and online updates.
"""

import numpy as np
from scipy.special import logsumexp


class MarkovJumpSystem:
    """Markov Jump System for modeling device state transitions.

    Models device operating modes as a discrete-time Markov chain with
    N=4 states: normal, degradation, warning, fault.
    """

    MODE_NORMAL = 0
    MODE_DEGRADATION = 1
    MODE_WARNING = 2
    MODE_FAULT = 3
    MODE_NAMES = ['Normal', 'Degradation', 'Warning', 'Fault']

    def __init__(self, n_modes=4):
        self.n_modes = n_modes
        # Initialize with uniform transition matrix
        self.Pi = np.ones((n_modes, n_modes)) / n_modes
        # Mode observation parameters: mean and covariance for each mode
        self.mode_means = None
        self.mode_covs = None
        self.initial_probs = np.ones(n_modes) / n_modes

    def initialize_constrained(self):
        """Initialize with physically meaningful transition probabilities.
        Normal states tend to persist, fault transitions are rare.
        """
        Pi = np.array([
            [0.85, 0.08, 0.05, 0.02],  # Normal: mostly stays normal
            [0.10, 0.70, 0.15, 0.05],  # Degradation: can recover or worsen
            [0.05, 0.10, 0.65, 0.20],  # Warning: likely to worsen
            [0.02, 0.03, 0.10, 0.85],  # Fault: tends to persist
        ])
        self.Pi = Pi
        self.initial_probs = np.array([0.7, 0.15, 0.10, 0.05])
        return self

    def estimate_mle(self, mode_sequence):
        """Estimate transition probabilities using MLE from observed mode sequence.

        Args:
            mode_sequence: array of mode labels (0 to n_modes-1)
        Returns:
            self
        """
        counts = np.zeros((self.n_modes, self.n_modes))
        for t in range(len(mode_sequence) - 1):
            i, j = mode_sequence[t], mode_sequence[t + 1]
            counts[i, j] += 1

        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        self.Pi = counts / row_sums
        return self

    def _forward(self, log_emission_probs):
        """Forward pass of forward-backward algorithm."""
        T = log_emission_probs.shape[0]
        log_alpha = np.zeros((T, self.n_modes))

        # Initialization
        log_alpha[0] = np.log(self.initial_probs + 1e-300) + log_emission_probs[0]

        # Recursion
        log_Pi = np.log(self.Pi + 1e-300)
        for t in range(1, T):
            for j in range(self.n_modes):
                log_alpha[t, j] = logsumexp(log_alpha[t-1] + log_Pi[:, j]) + log_emission_probs[t, j]

        return log_alpha

    def _backward(self, log_emission_probs):
        """Backward pass of forward-backward algorithm."""
        T = log_emission_probs.shape[0]
        log_beta = np.zeros((T, self.n_modes))

        # Initialization
        log_beta[-1] = 0.0

        # Recursion
        log_Pi = np.log(self.Pi + 1e-300)
        for t in range(T - 2, -1, -1):
            for i in range(self.n_modes):
                log_beta[t, i] = logsumexp(
                    log_Pi[i, :] + log_emission_probs[t + 1] + log_beta[t + 1]
                )

        return log_beta

    def _compute_emission_probs(self, features):
        """Compute log emission probabilities for each mode.

        Args:
            features: array of shape (T, d) - feature vectors
        Returns:
            log_emission_probs: array of shape (T, n_modes)
        """
        T = features.shape[0]
        log_emission = np.zeros((T, self.n_modes))

        for k in range(self.n_modes):
            mean = self.mode_means[k]
            cov = self.mode_covs[k]
            diff = features - mean
            try:
                inv_cov = np.linalg.inv(cov)
                sign, logdet = np.linalg.slogdet(cov)
                if sign <= 0:
                    logdet = np.log(np.linalg.det(cov + 1e-6 * np.eye(cov.shape[0])))
                d = features.shape[1]
                mahal = np.sum(diff @ inv_cov * diff, axis=1)
                log_emission[:, k] = -0.5 * (d * np.log(2 * np.pi) + logdet + mahal)
            except np.linalg.LinAlgError:
                log_emission[:, k] = -np.sum(diff ** 2, axis=1) / 2.0

        return log_emission

    def fit_em(self, features, max_iter=100, tol=1e-6):
        """Estimate parameters using EM algorithm.

        Args:
            features: array of shape (T, d) - feature vectors
            max_iter: maximum EM iterations
            tol: convergence tolerance
        Returns:
            self
        """
        T, d = features.shape

        # Initialize mode parameters using K-means-like initialization
        self._initialize_mode_params(features)

        prev_log_likelihood = -np.inf

        for iteration in range(max_iter):
            # E-step
            log_emission = self._compute_emission_probs(features)
            log_alpha = self._forward(log_emission)
            log_beta = self._backward(log_emission)

            # Gamma: posterior mode probabilities
            log_gamma = log_alpha + log_beta
            log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)

            # Xi: joint posterior of consecutive modes
            log_Pi = np.log(self.Pi + 1e-300)
            xi = np.zeros((T - 1, self.n_modes, self.n_modes))
            for t in range(T - 1):
                log_xi_t = (log_alpha[t, :, np.newaxis] + log_Pi +
                           log_emission[t + 1, np.newaxis, :] +
                           log_beta[t + 1, np.newaxis, :])
                log_xi_t -= logsumexp(log_xi_t)
                xi[t] = np.exp(log_xi_t)

            # Log-likelihood
            log_likelihood = logsumexp(log_alpha[-1])

            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < tol:
                break
            prev_log_likelihood = log_likelihood

            # M-step
            # Update initial probabilities
            self.initial_probs = gamma[0]

            # Update transition matrix
            xi_sum = xi.sum(axis=0)  # (n_modes, n_modes)
            row_sums = xi_sum.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            self.Pi = xi_sum / row_sums

            # Update emission parameters
            for k in range(self.n_modes):
                weight = gamma[:, k]
                weight_sum = weight.sum()
                if weight_sum > 1e-8:
                    self.mode_means[k] = (weight[:, np.newaxis] * features).sum(axis=0) / weight_sum
                    diff = features - self.mode_means[k]
                    weighted_diff = weight[:, np.newaxis] * diff
                    self.mode_covs[k] = (weighted_diff.T @ diff) / weight_sum
                    self.mode_covs[k] += 1e-6 * np.eye(d)  # regularization

        return self

    def _initialize_mode_params(self, features):
        """Initialize mode means and covariances using quantile-based splitting."""
        T, d = features.shape
        self.mode_means = np.zeros((self.n_modes, d))
        self.mode_covs = np.zeros((self.n_modes, d, d))

        # Sort by first principal component
        if d > 1:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            scores = pca.fit_transform(features).flatten()
        else:
            scores = features.flatten()

        sorted_idx = np.argsort(scores)
        segment_size = T // self.n_modes

        for k in range(self.n_modes):
            start = k * segment_size
            end = start + segment_size if k < self.n_modes - 1 else T
            segment_features = features[sorted_idx[start:end]]
            self.mode_means[k] = segment_features.mean(axis=0)
            self.mode_covs[k] = np.cov(segment_features.T) + 1e-6 * np.eye(d)
            if self.mode_covs[k].ndim == 0:  # 1D case
                self.mode_covs[k] = np.array([[self.mode_covs[k]]])

    def decode(self, features):
        """Viterbi decoding to find most likely mode sequence.

        Args:
            features: array of shape (T, d)
        Returns:
            modes: array of shape (T,) with mode labels
        """
        T = features.shape[0]
        log_emission = self._compute_emission_probs(features)
        log_Pi = np.log(self.Pi + 1e-300)
        log_init = np.log(self.initial_probs + 1e-300)

        # Viterbi
        delta = np.zeros((T, self.n_modes))
        psi = np.zeros((T, self.n_modes), dtype=int)

        delta[0] = log_init + log_emission[0]

        for t in range(1, T):
            for j in range(self.n_modes):
                scores = delta[t-1] + log_Pi[:, j]
                psi[t, j] = np.argmax(scores)
                delta[t, j] = scores[psi[t, j]] + log_emission[t, j]

        # Backtrace
        modes = np.zeros(T, dtype=int)
        modes[-1] = np.argmax(delta[-1])
        for t in range(T - 2, -1, -1):
            modes[t] = psi[t + 1, modes[t + 1]]

        return modes

    def online_update(self, new_counts, decay=0.1):
        """Online update of transition probabilities with decay factor.

        Args:
            new_counts: dict of (i,j) -> count from recent window
            decay: forgetting factor beta
        """
        new_matrix = np.zeros((self.n_modes, self.n_modes))
        for (i, j), count in new_counts.items():
            new_matrix[i, j] = count

        row_sums = new_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        new_probs = new_matrix / row_sums

        # Blend with old estimates
        mask = (row_sums.flatten() > 0)
        for i in range(self.n_modes):
            if mask[i]:
                self.Pi[i] = (1 - decay) * self.Pi[i] + decay * new_probs[i]

    def get_transition_matrix(self):
        """Return current transition probability matrix."""
        return self.Pi.copy()
