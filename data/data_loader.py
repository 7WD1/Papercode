"""Data Loading and Synthetic Data Generation for HRLAD"""

import numpy as np
import os


def generate_synthetic_ce_data(dataset_type='smartphone', n_samples=5000,
                                anomaly_ratio=0.05, seed=42, return_modes=False):
    """Generate synthetic consumer electronics time series data.

    Args:
        dataset_type: 'smartphone', 'appliance', or 'battery'
        n_samples: total number of time steps
        anomaly_ratio: fraction of anomalous points
        seed: random seed
        return_modes: if True, also return the underlying mode sequence
    Returns:
        values: 1D numpy array of time series values
        labels: 1D numpy array of anomaly labels (0=normal, 1=anomaly)
        modes (optional): 1D numpy array of mode labels (returned if return_modes=True)
    """
    rng = np.random.RandomState(seed)
    labels = np.zeros(n_samples, dtype=int)

    if dataset_type == 'smartphone':
        # Simulate smartphone sensor data (accelerometer-like)
        # Base signal: combination of periodic components
        t = np.linspace(0, n_samples / 50, n_samples)  # 50 Hz sampling
        values = (2.0 * np.sin(2 * np.pi * 0.5 * t) +  # slow wave
                  0.5 * np.sin(2 * np.pi * 2.0 * t) +   # medium wave
                  0.1 * rng.randn(n_samples))              # noise

        # Add mode transitions (device state changes)
        mode = np.zeros(n_samples, dtype=int)  # 0=normal
        transition_points = rng.choice(range(200, n_samples - 200),
                                       size=8, replace=False)
        transition_points.sort()

        current_mode = 0
        for tp in transition_points:
            duration = rng.randint(50, 200)
            end = min(tp + duration, n_samples)
            if rng.random() < 0.3:  # some transitions are anomalous
                mode[tp:end] = 3  # fault
                labels[tp:end] = 1
            elif rng.random() < 0.5:
                mode[tp:end] = 1  # degradation
            else:
                mode[tp:end] = 2  # warning

        # Add mode-specific signal modifications
        values[mode == 1] += 0.3 * rng.randn(np.sum(mode == 1))  # slight shift
        values[mode == 2] += 1.0 + 0.5 * rng.randn(np.sum(mode == 2))  # bigger shift
        values[mode == 3] += 3.0 + rng.randn(np.sum(mode == 3))  # large shift (anomaly)

        # Add point anomalies
        n_point_anomalies = int(n_samples * anomaly_ratio * 0.3)
        anomaly_idx = rng.choice(range(50, n_samples - 50),
                                 size=n_point_anomalies, replace=False)
        values[anomaly_idx] += rng.choice([-1, 1], size=n_point_anomalies) * rng.uniform(3, 6, n_point_anomalies)
        labels[anomaly_idx] = 1

        if return_modes:
            return values, labels, mode

    elif dataset_type == 'appliance':
        # Simulate home appliance power consumption
        t = np.linspace(0, n_samples / 6, n_samples)  # 10-min intervals

        # Base: daily cycle + weekly pattern
        daily = 100 * np.sin(2 * np.pi * t / 24) + 200
        weekly = 20 * np.sin(2 * np.pi * t / (24 * 7))
        noise = 5 * rng.randn(n_samples)
        values = daily + weekly + noise

        # Ensure non-negative
        values = np.maximum(values, 10)

        # Add anomalous power spikes (device malfunction)
        n_anomalies = int(n_samples * anomaly_ratio)
        anomaly_starts = rng.choice(range(100, n_samples - 100, 50),
                                     size=min(n_anomalies // 5, 20), replace=False)
        for start in anomaly_starts:
            duration = rng.randint(5, 30)
            end = min(start + duration, n_samples)
            spike = rng.uniform(150, 400)
            values[start:end] += spike
            labels[start:end] = 1

    elif dataset_type == 'battery':
        # Simulate battery management system data
        t = np.arange(n_samples)

        # Battery voltage pattern (3.0-4.2V range with discharge/charge cycles)
        cycle_length = 500
        voltage = np.zeros(n_samples)
        for i in range(n_samples):
            phase = (i % cycle_length) / cycle_length
            if phase < 0.7:  # discharging
                voltage[i] = 4.2 - 1.0 * (phase / 0.7)
            else:  # charging
                voltage[i] = 3.2 + 1.0 * ((phase - 0.7) / 0.3)

        temperature = 25 + 5 * np.sin(2 * np.pi * t / 200) + rng.randn(n_samples) * 0.5
        values = voltage + 0.1 * temperature + rng.randn(n_samples) * 0.05

        # Thermal runaway anomaly
        n_anomalies = int(n_samples * anomaly_ratio)
        anomaly_starts = rng.choice(range(100, n_samples - 100, 30),
                                     size=min(n_anomalies // 8, 15), replace=False)
        for start in anomaly_starts:
            duration = rng.randint(10, 50)
            end = min(start + duration, n_samples)
            # Voltage drop + temperature spike
            values[start:end] -= rng.uniform(0.5, 2.0)
            labels[start:end] = 1

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    if return_modes:
        return values, labels, np.zeros(n_samples, dtype=int)
    return values, labels


def load_nab_data(data_dir=None):
    """Load NAB dataset.

    If data_dir is provided, loads from local NAB data.
    Otherwise, generates synthetic data mimicking NAB characteristics.

    Returns:
        datasets: list of (name, values, labels) tuples
    """
    if data_dir and os.path.exists(data_dir):
        import pandas as pd
        datasets = []
        data_path = os.path.join(data_dir, 'data')
        labels_path = os.path.join(data_dir, 'labels')

        if os.path.exists(data_path):
            for root, dirs, files in os.walk(data_path):
                for f in files:
                    if f.endswith('.csv'):
                        filepath = os.path.join(root, f)
                        df = pd.read_csv(filepath)
                        if 'value' in df.columns:
                            values = df['value'].values.astype(float)
                            # Load corresponding labels
                            labels = np.zeros(len(values))
                            label_file = os.path.join(
                                labels_path,
                                os.path.relpath(root, data_path),
                                f.replace('.csv', '_labels.csv')
                            )
                            if os.path.exists(label_file):
                                label_df = pd.read_csv(label_file)
                                for _, row in label_df.iterrows():
                                    if row.get('label', 0) == 1:
                                        idx = (df['timestamp'] == row['timestamp']).values
                                        labels[idx] = 1
                            datasets.append((f, values, labels))

        if datasets:
            return datasets

    # Fallback: generate synthetic NAB-like data
    print("  Generating synthetic NAB-like data...")
    rng = np.random.RandomState(42)
    datasets = []

    for i in range(3):  # 3 different patterns
        n = 5000 + i * 1000
        t = np.arange(n)

        if i == 0:  # CPU usage pattern
            values = 50 + 15 * np.sin(2 * np.pi * t / 200) + rng.randn(n) * 3
        elif i == 1:  # Network traffic
            values = 100 + 30 * np.sin(2 * np.pi * t / 100) + rng.randn(n) * 10
        else:  # Temperature
            values = 40 + 8 * np.sin(2 * np.pi * t / 300) + rng.randn(n) * 2

        # Add anomalies
        labels = np.zeros(n)
        n_anomalies = rng.randint(3, 8)
        for _ in range(n_anomalies):
            start = rng.randint(200, n - 200)
            duration = rng.randint(10, 50)
            end = min(start + duration, n)
            values[start:end] += rng.uniform(30, 60)
            labels[start:end] = 1

        datasets.append((f'synthetic_nab_{i}', values, labels))

    return datasets


def load_yahoo_s5(data_dir=None):
    """Load Yahoo S5 dataset or generate synthetic equivalent.

    Returns:
        datasets: list of (name, values, labels) tuples
    """
    if data_dir and os.path.exists(data_dir):
        import pandas as pd
        datasets = []
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.endswith('.csv'):
                    filepath = os.path.join(root, f)
                    try:
                        df = pd.read_csv(filepath)
                        if 'value' in df.columns:
                            values = df['value'].values.astype(float)
                            labels = df.get('is_anomaly', df.get('anomaly',
                                       np.zeros(len(values)))).values.astype(int)
                            datasets.append((f, values, labels))
                    except Exception:
                        continue
        if datasets:
            return datasets

    # Synthetic Yahoo S5-like data
    print("  Generating synthetic Yahoo S5-like data...")
    rng = np.random.RandomState(42)
    datasets = []

    for group in range(4):  # A1-A4 groups
        n_series = 5 if group == 0 else 3
        for i in range(n_series):
            n = 1500
            t = np.arange(n)

            if group == 0:  # Real-like: complex patterns
                values = 100 + 20 * np.sin(2 * np.pi * t / 100) + rng.randn(n) * 5
            elif group == 1:  # Trend change
                values = np.cumsum(rng.randn(n) * 0.5) + 50
            elif group == 2:  # Mean/variance drift
                values = np.zeros(n)
                for j in range(n):
                    if j < n // 2:
                        values[j] = 50 + rng.randn() * 5
                    else:
                        values[j] = 70 + rng.randn() * 15
            else:  # Spikes
                values = 50 + rng.randn(n) * 3

            # Add anomalies
            labels = np.zeros(n)
            n_anom = rng.randint(2, 6)
            for _ in range(n_anom):
                pos = rng.randint(100, n - 100)
                if group == 3:  # point anomalies
                    values[pos] += rng.choice([-1, 1]) * rng.uniform(20, 50)
                    labels[pos] = 1
                else:
                    dur = rng.randint(5, 30)
                    end = min(pos + dur, n)
                    values[pos:end] += rng.uniform(15, 40)
                    labels[pos:end] = 1

            datasets.append((f'synthetic_yahoo_A{group+1}_{i}', values, labels))

    return datasets


def get_all_datasets(nab_dir=None, yahoo_dir=None):
    """Get all available datasets for evaluation.

    Returns:
        datasets: dict of dataset_name -> (values, labels)
    """
    all_datasets = {}

    # NAB datasets
    nab_data = load_nab_data(nab_dir)
    for name, values, labels in nab_data:
        all_datasets[f'NAB_{name}'] = (values, labels)

    # Yahoo S5 datasets
    yahoo_data = load_yahoo_s5(yahoo_dir)
    for name, values, labels in yahoo_data:
        all_datasets[f'Yahoo_{name}'] = (values, labels)

    # Synthetic consumer electronics datasets
    for ds_type, ds_name in [('smartphone', 'SPSD'),
                              ('appliance', 'HPMD'),
                              ('battery', 'BMSD')]:
        values, labels = generate_synthetic_ce_data(ds_type, n_samples=5000)
        all_datasets[ds_name] = (values, labels)

    return all_datasets
