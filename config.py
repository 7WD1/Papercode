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

    # === Dataset metadata (paper Section V-A and consolidated table tab:r1_datasets) ===
    # Every value below is copied verbatim from the manuscript; the synthetic
    # generators in data/data_loader.py read these so that the public code is
    # guaranteed to match the paper's dataset description.
    dataset_meta = {
        'SPSD': {
            'full_name': 'Smartphone Sensor Dataset',
            'domain': 'Smartphone sensors (accelerometer + gyroscope)',
            'sampling_rate_hz': 50,        # 50 Hz
            'n_channels': 6,               # 6 input channels (3 accel + 3 gyro)
            'channel_names': ['accel_x', 'accel_y', 'accel_z',
                              'gyro_x', 'gyro_y', 'gyro_z'],
            'selected_channel': 'accel_x', # paper L572 selection rule
            'devices': ['Samsung Galaxy S23', 'Xiaomi 13', 'OnePlus 11'],  # 3 phones
            'n_devices': 3,
            'anomaly_types': [             # 4 anomaly types
                'free-fall detection failure',
                'gyroscope drift',
                'accelerometer noise surge',
                'sensor fusion anomaly',
            ],
            'n_anomaly_types': 4,
            'n_anomaly_events': 1584,      # 1,584 annotated events
            'total_samples': 4_320_000,    # 4.32M effective samples
            'train_samples': 3_020_000,    # 3.02M
            'val_samples':   650_000,      # 0.65M
            'test_samples':  650_000,      # 0.65M
            'anomaly_ratio': 0.037,        # 3.7% (test)
            'monitoring_days': 30,
            'inter_annotator_kappa': 0.92, # Cohen's kappa > 0.92
        },
        'HPMD': {
            'full_name': 'Home Appliance Power Monitoring Dataset',
            'domain': 'Home appliance power monitoring',
            'sampling_rate_hz': 1,         # 1 Hz
            'n_channels': 3,               # 3 input channels (power of 3 appliances)
            'channel_names': ['refrigerator_power', 'ac_power', 'washer_power'],
            'selected_channel': 'refrigerator_power',  # = active power, paper L572
            'devices': ['LG GR-B247 refrigerator',
                        'Daikin FTXTA35 air conditioner',
                        'LG WD-T14410 washing machine'],  # 3 appliances
            'n_devices': 3,
            'anomaly_types': [             # 3 anomaly types
                'compressor start failure',
                'inverter abnormal oscillation',
                'motor stalling',
            ],
            'n_anomaly_types': 3,
            'n_anomaly_events': 1276,      # 1,276 annotated events
            'total_samples': 5_180_000,    # 5.18M effective samples
            'train_samples': 3_630_000,    # 3.63M
            'val_samples':   780_000,      # 0.78M
            'test_samples':  780_000,      # 0.78M
            'anomaly_ratio': 0.029,        # 2.9% (test)
            'monitoring_days': 60,
            'inter_annotator_kappa': 0.92,
        },
        'BMSD': {
            'full_name': 'Battery Management System Dataset',
            'domain': 'Battery management (V / I / T)',
            'sampling_rate_hz': 10,        # 10 Hz
            'n_channels': 3,               # 3 input channels (voltage, current, temperature)
            'channel_names': ['cell_voltage', 'pack_current', 'temperature'],
            'selected_channel': 'cell_voltage',  # paper L572
            'devices': ['4S2P lithium-polymer pack (14.8 V, 5000 mAh)'],  # 1 battery pack
            'n_devices': 1,
            'anomaly_types': [             # 3 anomaly types
                'overcharge / over-discharge',
                'thermal runaway precursor',
                'internal resistance degradation',
            ],
            'n_anomaly_types': 3,
            'n_anomaly_events': 1142,      # 1,142 annotated events
            'total_samples': 3_890_000,    # 3.89M effective samples
            'train_samples': 2_720_000,    # 2.72M
            'val_samples':   580_000,      # 0.58M
            'test_samples':  580_000,      # 0.58M
            'anomaly_ratio': 0.042,        # 4.2% (test)
            'monitoring_days': 45,
            'inter_annotator_kappa': 0.92,
        },
    }

    # === Injection-protocol realism parameters (paper Section V-F, response R1-C5) ===
    injection_gradual_onset_ms = (200, 800)   # fault magnitude ramped over 200-800 ms
    injection_partial_occlusion_ratio = 0.15  # one channel occluded during 15% of events

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
