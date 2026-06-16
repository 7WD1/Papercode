"""HRLAD Configuration - Hyperparameters and settings"""

class Config:
    # === Feature Extraction ===
    window_size = 64           # sliding window length w
    n_fft_components = 4       # number of FFT frequency components m
    feature_dim = 8            # 4 time-domain + 4 frequency-domain

    # === Markov Jump System ===
    n_modes = 4                # number of discrete modes: normal, degradation, warning, fault
    em_max_iter = 100          # EM algorithm max iterations
    em_tol = 1e-6              # EM convergence tolerance
    online_decay = 0.1         # online update decay factor beta

    # === Homotopy RL ===
    lambda_init = 0.0          # initial homotopy parameter
    lambda_step = 0.15         # homotopy step size delta_lambda
    lambda_max = 1.0           # maximum homotopy parameter
    gamma = 0.95               # discount factor
    alpha_reward = 1.0         # correct detection reward weight
    alpha_R = 1.0              # magnitude of the simplified reward R0
    beta_reward = 0.5          # false alarm penalty weight
    beta_w_reward = 0.2        # false warning penalty weight beta_w (paper Eq. reward)
    eta_reward = 0.1           # delay cost weight
    threshold_tau = 2.0        # initial threshold parameter tau
    sigmoid_alpha = 1.0        # sigmoid steepness for initial policy
    rho = 1.0                  # temperature for initial policy softmax/sigmoid scaling
    initial_policy_alpha = 1.0 # corresponds to alpha_rho = alpha / rho
    eta_adapt = 0.5            # adaptive scaling factor for online lambda increment
    delta_lambda_0 = 0.15      # initial lambda step size

    # === LSTD ===
    rbf_sigma = 1.0            # RBF kernel bandwidth
    lstd_reg = 1e-4            # regularization epsilon_r
    forgetting_factor = 0.97   # online forgetting factor zeta
    n_basis_functions = 20     # number of basis functions for LSTD

    # === Policy Iteration ===
    pi_max_iter = 50           # policy iteration max iterations
    pi_tol = 1e-3              # policy evaluation convergence threshold epsilon_V
    adaptive_step = True       # use adaptive lambda step

    # === Training ===
    offline_epochs = 10        # number of offline training epochs
    batch_size = 256           # batch size for training
    learning_rate = 1e-3       # learning rate

    # === Online Detection ===
    alarm_threshold = 0.5      # threshold for alarm decision
    tau_d = 0.5                # detection threshold for predict
    verification_delay = 50    # delayed-feedback steps delta (paper Section V-B)
    entropy_coef = 0.01        # actor entropy bonus alpha_H (paper Eq. actor_loss)

    # === Evaluation ===
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    # === Dataset names (paper Section V-A) ===
    datasets = ['SPSD', 'HPMD', 'BMSD']

    # === Random seed ===
    seed = 42

    # === Actor-Critic Network (Paper Table III: Network Architecture) ===
    actor_hidden_layers = [256, 128, 64]    # Actor: 3 layers [256, 128, 64]
    critic_hidden_layers = [256, 128, 64]   # Critic: 3 layers [256, 128, 64]
    activation = 'relu'                      # ReLU activation
    dropout_rate = 0.1                       # Dropout 0.1
    use_batch_norm = True                    # Batch normalization
    actor_lr = 3e-4                          # Actor learning rate
    critic_lr = 1e-3                         # Critic learning rate
    replay_buffer_size = 50000               # Replay buffer capacity
    target_update_freq = 5                   # Target update every 5 steps
    target_smooth_coeff = 0.005              # Polyak averaging tau_target
    epsilon_start = 1.0                      # Epsilon schedule start
    epsilon_end = 0.01                       # Epsilon schedule end
    epsilon_decay_episodes = 200             # Linear decay over 200 episodes

    # === Multi-seed Evaluation (Paper Section V) ===
    n_repeats = 10
    seeds = [42, 123, 456, 789, 2024, 314, 271, 1618, 999, 2048]
