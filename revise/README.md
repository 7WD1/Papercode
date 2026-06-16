# revise/ — representative data samples (supplementary)

Per the paper (IEEE TCE-2026-04-1642, Section V-A and the "Data and Code Availability" statement), this folder provides representative data samples as supplementary material: **10 normal and 10 anomalous windows per dataset, in CSV format**, for reviewer inspection.

## Contents

| Dataset | Normal windows | Anomalous windows | Window length | Channels |
|---------|:-:|:-:|:-:|:-:|
| SPSD | 10 | 10 | 64 | 6 (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z) |
| HPMD | 10 | 10 | 64 | 3 (refrigerator_power, ac_power, washer_power) |
| BMSD | 10 | 10 | 64 | 3 (cell_voltage, pack_current, temperature) |

## How generated

Acquired via the device-port interface (`data/device_ports.py`, mock replay) at the paper-declared sampling rates, then sliced into windows of length w=64 (paper Section V-B). The first paper seed (42) is used; re-running `python generate_revise_samples.py` reproduces every file exactly.

## Per-dataset details

- SPSD: normal=10 windows, anomalous=10 windows, channels=['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z'], selected=accel_x
- HPMD: normal=10 windows, anomalous=10 windows, channels=['refrigerator_power', 'ac_power', 'washer_power'], selected=refrigerator_power
- BMSD: normal=10 windows, anomalous=10 windows, channels=['cell_voltage', 'pack_current', 'temperature'], selected=cell_voltage
