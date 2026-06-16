"""Homotopic Reinforcement Learning Anomaly Detection (HRLAD) Module

Core algorithm implementing data-driven homotopic RL for time series
anomaly detection based on Markov Jump System framework.

Reference: Data-Driven Homotopic Reinforcement Learning-Based Adaptive
Optimal Control for Markov Jump Nonlinear Systems
"""

import numpy as np

from models.markov_jump import MarkovJumpSystem


class HRLADDetector:
    """HRLAD: Homotopic Reinforcement Learning Anomaly Detector.

    Implements the full HRLAD pipeline:
    1. Feature extraction + state space construction
    2. Markov jump transition estimation
    3. Homotopy RL with LSTD policy evaluation
    4. Online adaptive detection
    """

    # Action constants
    ACTION_NORMAL = 0
    ACTION_WARNING = 1
    ACTION_ALARM = 2
    ACTION_NAMES = ['normal', 'warning', 'alarm']
    N_ACTIONS = 3

    def __init__(self, config=None):
        if config is None:
            from config import Config
            config = Config()

        self.config = config
        self.window_size = config.window_size
        self.gamma = config.gamma
        self.lambda_val = config.lambda_init
        self.lambda_step = config.lambda_step
        self.alpha_reward = config.alpha_reward
        self.alpha_R = getattr(config, 'alpha_R', config.alpha_reward)
        self.beta_reward = config.beta_reward
        self.eta_reward = config.eta_reward
        self.threshold_tau = config.threshold_tau
        self.alpha_rho = getattr(config, 'initial_policy_alpha',
                                 getattr(config, 'sigmoid_alpha', 1.0))
        self.rho = getattr(config, 'rho', 1.0)
        self.rbf_sigma = config.rbf_sigma
        self.lstd_reg = config.lstd_reg
        self.forgetting_factor = config.forgetting_factor
        self.pi_tol = config.pi_tol
        self.pi_max_iter = config.pi_max_iter
        self.alarm_threshold = config.alarm_threshold
        self.tau_d = getattr(config, 'tau_d', 0.5)
        self.n_basis = config.n_basis_functions

        # Actor-Critic Network (optional, requires PyTorch)
        self.use_actor_critic = False
        self.ddpg_agent = None

        if getattr(config, 'actor_hidden_layers', None) is not None:
            try:
                import torch
                from models.actor_critic import DDPGAgent
                self.use_actor_critic = True
                self.ddpg_agent = DDPGAgent(
                    state_dim=config.feature_dim,
                    n_actions=self.N_ACTIONS,
                    actor_lr=config.actor_lr,
                    critic_lr=config.critic_lr,
                    tau=config.target_smooth_coeff,
                    buffer_capacity=config.replay_buffer_size,
                    target_update_freq=config.target_update_freq,
                    epsilon_start=config.epsilon_start,
                    epsilon_end=config.epsilon_end,
                    epsilon_decay_episodes=config.epsilon_decay_episodes,
                )
            except ImportError:
                pass

        # State: fitted parameters
        self.normal_mean_ = None
        self.normal_cov_ = None
        self.normal_cov_inv_ = None
        self.policy_weights_ = None
        self.basis_centers_ = None
        self.lstd_A_ = None
        self.lstd_b_ = None
        self.fitted = False

        # Markov Jump System
        self.mjs_ = None
        self.transition_matrix_ = None

        # Tracking
        self.lambda_history_ = []
        self.reward_history_ = []

    def _mahalanobis_sq(self, states):
        """Compute squared Mahalanobis distance from normal mean."""
        if self.normal_mean_ is None or self.normal_cov_inv_ is None:
            raise RuntimeError("Normal statistics not computed.")
        diff = states - self.normal_mean_
        return np.sum(diff * (diff @ self.normal_cov_inv_), axis=1)

    def _rbf_features(self, x, centers):
        """Compute RBF basis function features.

        Args:
            x: state vector(s), shape (d,) or (n, d)
            centers: basis centers, shape (K, d)
        Returns:
            phi: basis features, shape (K,) or (n, K)
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Pairwise squared distances
        diff = x[:, np.newaxis, :] - centers[np.newaxis, :, :]
        sq_dist = np.sum(diff ** 2, axis=2)

        phi = np.exp(-sq_dist / (2 * self.rbf_sigma ** 2))
        return phi.squeeze()

    def _kernel_matrix(self, states, next_states):
        """Compute N x N Gaussian RBF kernel matrix with Mahalanobis distance.

        k(x_i, x_j) = exp(-0.5 * (x_i - x_j)^T Sigma_0^{-1} (x_i - x_j) / sigma_k^2)
        """
        # states shape (N, d), next_states shape (N, d)
        diff = states[:, np.newaxis, :] - next_states[np.newaxis, :, :]
        # Mahalanobis squared distance: sum over d of diff * (diff @ Sigma_0^{-1})
        mahal_sq = np.sum(diff * (diff @ self.normal_cov_inv_), axis=2)
        K = np.exp(-0.5 * mahal_sq / (self.rbf_sigma ** 2))
        return K

    def _kernel_vector(self, x, train_states):
        """Compute kernel vector k(x) against training states.

        k(x) = [k(x, x_1), ..., k(x, x_N)]^T
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        diff = x[:, np.newaxis, :] - train_states[np.newaxis, :, :]
        mahal_sq = np.sum(diff * (diff @ self.normal_cov_inv_), axis=2)
        k_vec = np.exp(-0.5 * mahal_sq / (self.rbf_sigma ** 2))
        return k_vec.squeeze()

    def _compute_initial_policy(self, states):
        """Compute initial softmax policy pi_0 (Paper eq. 543-544, 548-549, 560).

        pi_0(a|x) = exp(rho^-1 * Q_0(x,a;xi_0)) /
                     sum_{a'} exp(rho^-1 * Q_0(x,a';xi_0))

        Where Q_0 is the simplified action-value function (Paper eq. 560):
        - Q_0(alarm, x)   = alpha_R * sign(d_M^2(x, x_normal; Sigma_0) - tau)
        - Q_0(other, x)   = 0  (R_0 only defined for alarm action)
        """
        mahal = self._mahalanobis_sq(states)

        # Simplified Q-values for each action (Paper eq:simplified_reward)
        # R_0 is only defined for alarm action; other actions get Q=0
        Q_normal = np.zeros(len(states))
        Q_warning = np.zeros(len(states))  # not defined in paper
        Q_alarm = self.alpha_R * np.sign(mahal - self.threshold_tau)
        Q = np.column_stack([Q_normal, Q_warning, Q_alarm])

        # Softmax with temperature rho (paper eq. 563)
        Q_scaled = Q / self.rho
        Q_scaled -= Q_scaled.max(axis=1, keepdims=True)
        exp_Q = np.exp(Q_scaled)
        probs = exp_Q / exp_Q.sum(axis=1, keepdims=True)
        return probs

    def _mode_to_action(self, mode_label):
        """Map mode label to ground-truth action (Paper eq. 315).

        Modes: 0=normal, 1=degradation, 2=warning, 3=fault
        Actions: 0=normal, 1=warning, 2=alarm
        """
        if mode_label == 0:  # normal -> normal action
            return self.ACTION_NORMAL
        elif mode_label == 3:  # fault -> alarm action
            return self.ACTION_ALARM
        else:  # degradation(1), warning(2) -> warning action
            return self.ACTION_WARNING

    def _compute_reward(self, state, action, true_label, delay=0, mode_label=None):
        """Compute the homotopy-parameterized reward (Paper eq. 574, 315).

        R_lambda(x, a) = (1-lambda) * R_0(x, a; xi_0) + lambda * R(x, a; xi*)
        """
        mahal = self._mahalanobis_sq(state.reshape(1, -1))[0]

        # Simplified reward R_0 (Paper eq. 579)
        if action == self.ACTION_ALARM:
            r0 = self.alpha_R * np.sign(mahal - self.threshold_tau)
        else:
            r0 = 0.0

        # Full reward R (Paper eq. 315)
        if mode_label is not None:
            correct_action = self._mode_to_action(mode_label)
        else:
            # Fallback to binary label: 1=anomaly->alarm, 0=normal->normal
            correct_action = self.ACTION_ALARM if true_label == 1 else self.ACTION_NORMAL

        if action == correct_action:
            r_full = self.alpha_reward  # +α for correct action (Paper eq:reward)
        elif action == self.ACTION_ALARM and correct_action == self.ACTION_NORMAL:
            r_full = -self.beta_reward  # -β for false alarm (Paper eq:reward)
        else:
            r_full = 0.0  # no explicit reward/penalty per paper

        r_full -= self.eta_reward * delay

        # Homotopy blend (Paper eq. 574)
        reward = (1 - self.lambda_val) * r0 + self.lambda_val * r_full
        return reward

    def _compute_batch_rewards(self, states, actions, labels, mode_labels=None):
        """Compute rewards for a batch of transitions."""
        rewards = np.array([
            self._compute_reward(
                states[i], actions[i], labels[i],
                mode_label=mode_labels[i] if mode_labels is not None else None
            )
            for i in range(len(states))
        ])
        return rewards

    def _lstd_solve(self, states, rewards, next_states):
        """Solve kernelized LSTD for value function coefficients.

        Computes alpha = (K - gamma*K' + lstd_reg*I_N)^{-1} @ R_lambda
        where K_ij = k(x_i, x_j) and K'_ij = k(x_i, x'_j).
        Falls back to old RBF basis method if N > 8000 for memory safety.
        """
        N = len(states)
        if N > 8000:
            # Fallback to RBF basis to avoid OOM
            phi = self._rbf_features(states, self.basis_centers_)
            phi_next = self._rbf_features(next_states, self.basis_centers_)
            if phi.ndim == 1:
                phi = phi.reshape(1, -1)
            if phi_next.ndim == 1:
                phi_next = phi_next.reshape(1, -1)
            Kb = phi.shape[1]
            A = np.zeros((Kb, Kb))
            b_vec = np.zeros(Kb)
            for i in range(len(phi)):
                A += np.outer(phi[i], phi[i] - self.gamma * phi_next[i])
                b_vec += phi[i] * rewards[i]
            A += self.lstd_reg * np.eye(Kb)
            try:
                theta = np.linalg.solve(A, b_vec)
            except np.linalg.LinAlgError:
                theta = np.linalg.lstsq(A, b_vec, rcond=None)[0]
            return theta

        # Use precomputed inverse if available (same states/next_states)
        if (hasattr(self, 'lstd_A_inv_') and self.lstd_A_inv_.shape[0] == N):
            return self.lstd_A_inv_ @ rewards

        K = self._kernel_matrix(states, states)
        Kp = self._kernel_matrix(states, next_states)
        A = K - self.gamma * Kp + self.lstd_reg * np.eye(N)
        try:
            alpha = np.linalg.solve(A, rewards)
        except np.linalg.LinAlgError:
            alpha = np.linalg.lstsq(A, rewards, rcond=None)[0]
        return alpha

    def _compute_greedy_policy(self, states, next_states, state_labels,
                                state_modes=None):
        """Policy improvement: deterministic greedy policy pi'.

        For each state x_i, compute one-step lookahead Q-values:
            Q(x_i, a) = R_lambda(x_i, a) + gamma * E[V(x_{i+1})]
        where the expectation over modes uses the Markov transition matrix:
            E[V(x_{i+1})] = sum_{s'} Pi[mode_i, s'] * V(x_{i+1})
        Then set pi'(a|x_i) = 1 if a == argmax_a' Q(x_i, a'), else 0.
        """
        if self.policy_weights_ is None:
            n = len(states)
            greedy_probs = np.zeros((n, self.N_ACTIONS))
            greedy_probs[:, self.ACTION_NORMAL] = 1.0
            return greedy_probs, np.zeros(n, dtype=int)

        if (hasattr(self, 'K_train_next_') and
                len(next_states) == len(self.train_next_states_) and
                len(self.policy_weights_) == len(self.train_states_)):
            V_next = self.K_train_next_ @ self.policy_weights_
        elif (hasattr(self, 'train_states_') and
                len(self.policy_weights_) == len(self.train_states_)):
            k_next = self._kernel_vector(next_states, self.train_states_)
            if k_next.ndim == 1:
                k_next = k_next.reshape(1, -1)
            V_next = k_next @ self.policy_weights_
        else:
            phi_next = self._rbf_features(next_states, self.basis_centers_)
            if phi_next.ndim == 1:
                phi_next = phi_next.reshape(1, -1)
            V_next = phi_next @ self.policy_weights_

        # MJS-aware expected next-state value
        if (state_modes is not None and self.transition_matrix_ is not None
                and len(state_modes) == len(states)):
            expected_V_next = np.zeros(len(states))
            for i in range(len(states)):
                mode_i = state_modes[i]
                expected_V_next[i] = np.sum(
                    self.transition_matrix_[mode_i] * V_next[i]
                )
        else:
            expected_V_next = V_next

        n = len(states)
        greedy_actions = np.zeros(n, dtype=int)

        for i in range(n):
            q_values = np.zeros(self.N_ACTIONS)
            for a in range(self.N_ACTIONS):
                r = self._compute_reward(
                    states[i], a, state_labels[i],
                    mode_label=state_modes[i] if state_modes is not None else None
                )
                q_values[a] = r + self.gamma * expected_V_next[i]
            greedy_actions[i] = int(np.argmax(q_values))

        greedy_probs = np.zeros((n, self.N_ACTIONS))
        greedy_probs[np.arange(n), greedy_actions] = 1.0
        return greedy_probs, greedy_actions

    def fit(self, features, labels, verbose=True):
        """Offline training: homotopy RL with policy iteration.

        Args:
            features: normalized feature array of shape (T, d)
            labels: anomaly labels of shape (T,) (0=normal, 1=anomaly)
            verbose: print progress
        Returns:
            self
        """
        n_samples = len(features)

        # Compute normal mean and covariance from normal-labeled samples
        normal_mask = labels == 0
        if normal_mask.sum() > 0:
            normal_data = features[normal_mask]
            self.normal_mean_ = normal_data.mean(axis=0)
            if normal_data.shape[0] > 1:
                self.normal_cov_ = np.cov(normal_data, rowvar=False)
            else:
                d = normal_data.shape[1] if normal_data.ndim > 1 else 1
                self.normal_cov_ = np.eye(d) * 1e-3
        else:
            self.normal_mean_ = features.mean(axis=0)
            if features.shape[0] > 1:
                self.normal_cov_ = np.cov(features, rowvar=False)
            else:
                d = features.shape[1] if features.ndim > 1 else 1
                self.normal_cov_ = np.eye(d) * 1e-3

        # Ensure 2D covariance matrix
        if self.normal_cov_.ndim < 2:
            self.normal_cov_ = np.atleast_2d(self.normal_cov_)
        if self.normal_cov_.shape[0] != self.normal_cov_.shape[1]:
            d = self.normal_mean_.shape[0]
            self.normal_cov_ = np.eye(d) * 1e-3

        # Safe inversion
        try:
            self.normal_cov_inv_ = np.linalg.inv(self.normal_cov_)
        except np.linalg.LinAlgError:
            d = self.normal_cov_.shape[0]
            self.normal_cov_inv_ = np.linalg.inv(self.normal_cov_ + 1e-6 * np.eye(d))

        # Select basis function centers (kept for compatibility / fallback)
        n_centers = min(self.n_basis, n_samples)
        center_idx = np.random.choice(n_samples, size=n_centers, replace=False)
        self.basis_centers_ = features[center_idx]

        # Estimate Markov Jump System parameters
        self.mjs_ = MarkovJumpSystem(n_modes=self.config.n_modes)
        self.mjs_.initialize_constrained()
        self.mjs_.fit_em(features, max_iter=self.config.em_max_iter,
                         tol=self.config.em_tol)
        modes = self.mjs_.decode(features)
        self.transition_matrix_ = self.mjs_.get_transition_matrix()
        state_modes = modes[:-1]

        # Prepare transition data
        states = features[:-1]
        next_states = features[1:]
        state_labels = labels[1:]

        # Save training states for kernel evaluations
        self.train_states_ = states
        self.train_next_states_ = next_states

        # Precompute fixed kernel LSTD matrix inverse (expensive, done once)
        N = len(states)
        if N <= 8000:
            self.K_train_self_ = self._kernel_matrix(states, states)
            self.K_train_next_ = self._kernel_matrix(states, next_states)
            A_fixed = self.K_train_self_ - self.gamma * self.K_train_next_ + self.lstd_reg * np.eye(N)
            try:
                self.lstd_A_inv_ = np.linalg.inv(A_fixed)
            except np.linalg.LinAlgError:
                self.lstd_A_inv_ = np.linalg.inv(A_fixed + 1e-6 * np.eye(N))

        # Online basis centers: fixed smaller set (e.g., 20 uniformly selected)
        n_online = min(20, len(states))
        online_idx = np.linspace(0, len(states) - 1, n_online, dtype=int)
        self.online_basis_centers_ = states[online_idx]

        # Initialize policy from pi_0
        policy_probs = self._compute_initial_policy(states)
        actions = np.array([
            np.random.choice(self.N_ACTIONS, p=policy_probs[i])
            for i in range(len(states))
        ])

        # === Homotopy continuation ===
        self.lambda_val = self.config.lambda_init
        step = self.lambda_step

        while self.lambda_val < self.config.lambda_max:
            rewards = self._compute_batch_rewards(states, actions, state_labels,
                                                   mode_labels=state_modes)
            values_history = []
            converged = False

            # Policy iteration for current lambda
            for pi_iter in range(self.pi_max_iter):
                # Policy evaluation (LSTD)
                self.policy_weights_ = self._lstd_solve(states, rewards, next_states)

                if (hasattr(self, 'K_train_self_') and
                        len(self.policy_weights_) == len(self.train_states_)):
                    values = self.K_train_self_ @ self.policy_weights_
                elif (hasattr(self, 'train_states_') and
                        len(self.policy_weights_) == len(self.train_states_)):
                    k_vec = self._kernel_vector(states, self.train_states_)
                    if k_vec.ndim == 1:
                        k_vec = k_vec.reshape(1, -1)
                    values = k_vec @ self.policy_weights_
                else:
                    phi = self._rbf_features(states, self.basis_centers_)
                    if phi.ndim == 1:
                        phi = phi.reshape(1, -1)
                    values = phi @ self.policy_weights_
                values_history.append(values)

                # Policy improvement: greedy Q-based with MJS transition model
                greedy_probs, greedy_actions = self._compute_greedy_policy(
                    states, next_states, state_labels, state_modes=state_modes
                )

                # Re-sample actions from the greedy policy
                actions = greedy_actions.copy()
                rewards = self._compute_batch_rewards(states, actions, state_labels,
                                                       mode_labels=state_modes)

                # Convergence check: rho = ||V_new - V_old||_inf (eq. 715)
                if len(values_history) >= 2:
                    rho = np.max(np.abs(values_history[-1] - values_history[-2]))
                    if rho < self.pi_tol:
                        converged = True
                        if verbose:
                            print(f"    PI converged at iter {pi_iter+1}, "
                                  f"lambda={self.lambda_val:.4f}, rho={rho:.6f}")
                        break

            avg_reward = np.mean(rewards)
            self.lambda_history_.append(self.lambda_val)
            self.reward_history_.append(avg_reward)

            if verbose:
                print(f"  lambda={self.lambda_val:.4f}, "
                      f"avg_reward={avg_reward:.4f}, converged={converged}")

            # Increment lambda when PI converges (rho < pi_tol); otherwise
            # still step with a reduced step to avoid deadlock.
            if converged:
                if self.config.adaptive_step and len(self.reward_history_) >= 2:
                    improvement = self.reward_history_[-1] - self.reward_history_[-2]
                    if improvement > 0.01:
                        step = min(step * 1.5, 0.2)
                    elif improvement < -0.01:
                        step = max(step * 0.5, 0.02)
                self.lambda_val = min(self.lambda_val + step, self.config.lambda_max)
            else:
                step = max(step * 0.5, 0.02)
                self.lambda_val = min(self.lambda_val + step, self.config.lambda_max)

        # Actor-Critic training after homotopy loop completes
        if self.use_actor_critic and self.ddpg_agent is not None:
            import torch
            if verbose:
                print("  Training actor-critic network...")
            for ep in range(max(1, self.config.offline_epochs)):
                for i in range(0, len(states), self.config.batch_size):
                    batch_states = states[i:i+self.config.batch_size]
                    batch_actions = actions[i:i+self.config.batch_size]
                    batch_rewards = rewards[i:i+self.config.batch_size]
                    batch_next = next_states[i:i+self.config.batch_size]

                    # Push to replay buffer and update
                    for j in range(len(batch_states)):
                        self.ddpg_agent.replay_buffer.push(
                            torch.FloatTensor(batch_states[j]),
                            int(batch_actions[j]),
                            float(batch_rewards[j]),
                            torch.FloatTensor(batch_next[j]),
                            False
                        )

                    if len(self.ddpg_agent.replay_buffer) >= self.config.batch_size:
                        self.ddpg_agent.update(self.config.batch_size)
            if verbose:
                print(f"  Actor-critic training complete ({ep+1} epochs)")

        # Initialize online LSTD matrices
        p = len(self.online_basis_centers_)
        self.lstd_A_ = self.lstd_reg * np.eye(p)
        self.lstd_b_ = np.zeros(p)
        self.online_theta_ = np.zeros(p)
        # Precompute online basis features for training states
        self.online_phi_train_ = self._rbf_features(states, self.online_basis_centers_)
        if self.online_phi_train_.ndim == 1:
            self.online_phi_train_ = self.online_phi_train_.reshape(1, -1)
        # Cache values for online lambda adaptation (kernel-based initial values)
        if (hasattr(self, 'train_states_') and
                len(self.policy_weights_) == len(self.train_states_)):
            k_vec = self._kernel_vector(states, self.train_states_)
            if k_vec.ndim == 1:
                k_vec = k_vec.reshape(1, -1)
            self.online_values_ = (k_vec @ self.policy_weights_).squeeze()
        else:
            phi = self._rbf_features(states, self.basis_centers_)
            if phi.ndim == 1:
                phi = phi.reshape(1, -1)
            self.online_values_ = (phi @ self.policy_weights_).squeeze()

        self.fitted = True
        return self

    def predict(self, features):
        """Predict anomaly labels for feature sequences.

        Args:
            features: normalized feature array of shape (T, d)
        Returns:
            predictions: binary predictions (0=normal, 1=anomaly)
            scores: anomaly scores = pi_lambda(a_alarm|x) (eq. 859)
        """
        if not self.fitted:
            raise RuntimeError("Must call fit() before predict()")

        n_samples = len(features)

        # Initial policy pi_0
        pi_0 = self._compute_initial_policy(features)

        # Use actor-critic network when available for improved policy
        if self.use_actor_critic and self.ddpg_agent is not None:
            import torch
            with torch.no_grad():
                x = torch.FloatTensor(features)
                if x.ndim == 1:
                    x = x.unsqueeze(0)
                actor_probs = self.ddpg_agent.actor(x)
                pi_prime = actor_probs.cpu().numpy()
        else:
            # Greedy policy pi' evaluated on the given features.
            # Use pi_0 alarm probabilities to generate pseudo-labels for the
            # reward computation so the greedy step remains label-free at test time.
            pseudo_labels = (pi_0[:, self.ACTION_ALARM] >= self.tau_d).astype(int)
            if n_samples > 1:
                next_features = np.vstack([features[1:], features[-1:]])
            else:
                next_features = features.copy()

            pi_prime, _ = self._compute_greedy_policy(features, next_features, pseudo_labels)

        # Full blended policy pi_lambda (convex combination of pi_0 and pi')
        pi_lambda = (1 - self.lambda_val) * pi_0 + self.lambda_val * pi_prime

        # Extract alarm probability as score
        scores = pi_lambda[:, self.ACTION_ALARM]

        # Detection output: y_hat = 1 if pi_lambda(a_alarm|x) >= tau_d
        predictions = (scores >= self.tau_d).astype(int)

        return predictions, scores

    def online_update(self, state, action, reward, next_state,
                      mode_transition_counts=None):
        """Online incremental update with forgetting factor (eq. 830--831).

        A_t = zeta * A_{t-1} + (1-zeta) * [phi(x)(phi(x) - gamma*phi(x'))^T + eps_reg*I_p]
        b_t = zeta * b_{t-1} + (1-zeta) * phi(x) * R_lambda(x, a)

        Optionally updates the Markov transition matrix if
        mode_transition_counts is provided.
        """
        if self.online_basis_centers_ is None:
            return

        phi = self._rbf_features(state, self.online_basis_centers_)
        phi_next = self._rbf_features(next_state, self.online_basis_centers_)

        zeta = self.forgetting_factor
        self.lstd_A_ = (zeta * self.lstd_A_ +
                        (1.0 - zeta) * (np.outer(phi, phi - self.gamma * phi_next) +
                                        self.lstd_reg * np.eye(len(self.online_basis_centers_))))
        self.lstd_b_ = (zeta * self.lstd_b_ +
                        (1.0 - zeta) * phi * reward)

        try:
            A_reg = self.lstd_A_ + self.lstd_reg * np.eye(len(self.online_basis_centers_))
            self.online_theta_ = np.linalg.solve(A_reg, self.lstd_b_)
        except np.linalg.LinAlgError:
            pass

        # Online lambda adaptation using value vectors over training states
        if hasattr(self, 'online_phi_train_') and hasattr(self, 'online_values_'):
            V_new = self.online_phi_train_ @ self.online_theta_
            if V_new.ndim > 1:
                V_new = V_new.squeeze()
            self._adapt_lambda_online(self.online_values_, V_new)
            self.online_values_ = V_new.copy()

        if mode_transition_counts is not None and self.mjs_ is not None:
            self.mjs_.online_update(mode_transition_counts,
                                    decay=self.config.online_decay)

    def _adapt_lambda_online(self, V_old, V_new):
        """Adaptive homotopy parameter increment (eq. 849)."""
        rho = np.max(np.abs(V_new - V_old))
        if rho < self.pi_tol and self.lambda_val < 1.0:
            delta_lambda = (self.config.delta_lambda_0 *
                            min(2.0, self.config.eta_adapt / (rho + self.pi_tol)))
            self.lambda_val = min(self.lambda_val + delta_lambda, 1.0)

    def get_anomaly_scores(self, features):
        """Get continuous anomaly scores for features."""
        _, scores = self.predict(features)
        return scores
