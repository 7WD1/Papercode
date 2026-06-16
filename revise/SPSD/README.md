# SPSD representative samples

Supplementary material for IEEE TCE-2026-04-1642 (paper Section V-A and the
"Data and Code Availability" statement).

## Source device

Smartphone Sensor Dataset — acquired from Samsung Galaxy S23, Xiaomi 13, OnePlus 11 via the
`SPSDDevicePort` interface (`data/device_ports.py`).

- Sampling rate: 50 Hz
- Input channels (6): accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
- Paper-selected channel (main.tex L572): **accel_x**
- Monitoring duration in the full dataset: 30 days
- Anomaly ratio (full test set): 3.7%

## Files

- `normal.csv`    — 10 normal windows, each 64 samples long.
- `anomalous.csv` — 10 anomalous windows, each 64 samples long.

## CSV format (long)

| column | meaning |
|--------|---------|
| `window_id` | window index 0-9 |
| `t` | sample index within the window (0..63) |
| `timestamp` | absolute time (s) = (global_sample_index) / sampling_rate |
| ``accel_x` | channel value
`accel_y` | channel value
`accel_z` | channel value
`gyro_x` | channel value
`gyro_y` | channel value
`gyro_z` | channel value |
| `label` | 0 = normal, 1 = anomaly (binary, paper Table I) |
| `mode` | latent mode 0=normal, 1=degradation, 2=warning, 3=fault |

The first channel column `accel_x` is the paper-selected best channel used
to produce the main results. The remaining channels are the full multivariate
device stream for reviewer inspection.

## Reproducibility

Windows were extracted with the fixed seed 42 (first of the paper's
10-seed protocol [42, 123, 456, 789, 2024, 314, 271, 1618, 999, 2048]). Re-running
`python generate_revise_samples.py` reproduces these files exactly because the
mock replay is deterministic for a given seed.
