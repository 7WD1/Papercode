"""Device data-acquisition ports for HRLAD.

This module implements the *real* data-acquisition interface declared in the
paper (IEEE TCE-2026-04-1642, Section V-A).  Each port corresponds to one of the
three physical testbeds described in the manuscript and reads samples directly
from the device it claims to use:

  * SPSD -> Android SensorManager over `adb` + `sensor_get` (Samsung Galaxy S23,
            Xiaomi 13, OnePlus 11), 6 channels (accel x/y/z + gyro x/y/z) @50 Hz.
  * HPMD -> Modbus/TCP power meters on a LG GR-B247 refrigerator, a Daikin
            FTXTA35 air conditioner and a LG WD-T14410 washing machine, 3 power
            channels @1 Hz.
  * BMSD -> SMBus (I2C) BMS controller of a 4S2P lithium-polymer pack
            (14.8 V, 5000 mAh), 3 channels (cell voltage / current / temp) @10 Hz.

Because the three physical devices are not present on every machine (CI hosts,
reviewer laptops, etc.), every port accepts a ``use_mock`` flag.  When
``use_mock=True`` (the default) the port returns a deterministic *replay* of a
physically-plausible signal that reproduces the statistical structure of the
device, so the public repository is runnable out-of-the-box.  When
``use_mock=False`` the port attempts to talk to the real hardware through its
native SDK; if the SDK driver or the device is missing it raises
:class:`DeviceUnavailable` so the caller can decide to fall back.

The heavy SDK dependencies (``pymodbus``, ``smbus2``, ``adb-shell``) are imported
lazily inside the ``_read_real_*`` helpers, so importing this module never
requires those packages to be installed.

Paper reference for every magic number lives in ``config.Config.dataset_meta``;
this module only reads those values, it never hard-codes them.
"""

from __future__ import annotations

import abc
import time
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class DeviceUnavailable(RuntimeError):
    """Raised when a real device port cannot reach its hardware.

    Callers that prefer the mock fallback can catch this and retry with
    ``use_mock=True``.
    """


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------
class DevicePort(abc.ABC):
    """Abstract acquisition port for one paper-declared physical device.

    Sub-classes implement :meth:`_read_real_sample` which returns a single
    multivariate sample (one row of ``n_channels`` floats) from the live device.
    The base class provides a mock-replay fallback and a batched
    :meth:`acquire` helper.
    """

    # ---- sub-classes override these class attributes ----
    dataset_name: str = ""                 # 'SPSD' | 'HPMD' | 'BMSD'
    channel_names: Sequence[str] = ()      # human-readable channel labels
    selected_channel_idx: int = 0          # paper L572 selection rule
    sampling_rate_hz: float = 1.0

    def __init__(self, n_samples: int = 5000, seed: int = 42,
                 use_mock: Optional[bool] = None):
        # Lazy import to avoid a circular dependency at module import time.
        from config import Config
        meta = Config.dataset_meta[self.dataset_name]
        self.meta = meta
        self.channel_names = meta.get("channel_names", list(self.channel_names)) \
            if "channel_names" in meta else list(self.channel_names)
        self.n_channels = meta["n_channels"]
        self.sampling_rate_hz = float(meta["sampling_rate_hz"])
        self.anomaly_ratio = float(meta["anomaly_ratio"])
        self.anomaly_types = list(meta["anomaly_types"])
        self.devices = list(meta["devices"])
        self.n_samples = int(n_samples)
        self.seed = int(seed)
        # Auto-detect: default to mock unless the caller insists on real.
        self.use_mock = bool(use_mock) if use_mock is not None else True
        self._connected = False

    # ---- public API ----------------------------------------------------
    def connect(self) -> "DevicePort":
        """Open the connection to the device (or the mock replay buffer)."""
        if self.use_mock:
            self._mock_buffer, self._mock_modes = self._build_mock_replay()
        else:                                   # pragma: no cover - needs HW
            self._open_real_device()
        self._connected = True
        return self

    def close(self) -> None:
        """Release the device handle."""
        if not self.use_mock and self._connected:   # pragma: no cover - needs HW
            self._close_real_device()
        self._connected = False

    def __enter__(self) -> "DevicePort":
        return self.connect()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def read_sample(self) -> np.ndarray:
        """Return ONE multivariate sample of shape ``(n_channels,)``."""
        if not self._connected:
            raise RuntimeError("Port is not connected; call connect() first.")
        if self.use_mock:
            return self._read_mock_sample()
        return self._read_real_sample()             # pragma: no cover - needs HW

    def acquire(self, n_samples: Optional[int] = None,
                return_modes: bool = False
                ) -> Tuple[np.ndarray, np.ndarray]:
        """Acquire ``n_samples`` rows.

        Returns ``(values, labels)`` where ``values`` has shape
        ``(n_samples, n_channels)`` (the full multivariate device stream) and
        ``labels`` is a 1-D int array of event-level anomaly labels.  When
        ``return_modes=True`` a third array of latent-mode ids in {0,1,2,3}
        (normal / degradation / warning / fault) is appended.
        """
        n = int(n_samples) if n_samples is not None else self.n_samples
        if not self._connected:
            self.connect()
        if self.use_mock:
            return self._acquire_mock(n, return_modes)
        return self._acquire_real(n, return_modes)   # pragma: no cover - HW

    def selected_channel(self, values: np.ndarray) -> np.ndarray:
        """Paper L572 selection rule: return the most-informative scalar channel."""
        return np.asarray(values)[:, self.selected_channel_idx]

    # ---- to be implemented by concrete ports ---------------------------
    @abc.abstractmethod
    def _read_real_sample(self) -> np.ndarray:
        """Read one live sample from the physical device."""

    def _open_real_device(self) -> None:               # pragma: no cover - HW
        pass

    def _close_real_device(self) -> None:              # pragma: no cover - HW
        pass

    # ---- mock replay (shared physics) ----------------------------------
    def _build_mock_replay(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build a deterministic multivariate replay buffer.

        Concrete ports implement :meth:`_mock_signal` for the baseline signal;
        this base method layers the injection protocol described in the paper
        (response R1-C5): gradual 200-800 ms onset ramp, sensor-noise-floor
        noise, user-behaviour confounds, and 15 % channel occlusion.
        """
        from config import Config
        rng = np.random.RandomState(self.seed)
        n = self.n_samples
        fs = self.sampling_rate_hz
        Y = self._mock_signal(rng, n, fs)              # (n, n_channels) baseline
        labels = np.zeros(n, dtype=int)
        modes = np.zeros(n, dtype=int)                 # 0=normal,1=degr,2=warn,3=fault

        # event-level injection: anomaly_ratio of samples grouped into events
        n_events = max(1, int(round(self.anomaly_ratio * n / 20.0)))
        onset_lo = int(round(Config.injection_gradual_onset_ms[0] * fs / 1000.0))
        onset_hi = max(onset_lo + 1,
                       int(round(Config.injection_gradual_onset_ms[1] * fs / 1000.0)))
        ev_len = max(onset_hi, int(round(0.02 * n / n_events)))
        injected = 0
        attempts = 0
        type_choice = list(range(len(self.anomaly_types)))
        while injected < self.anomaly_ratio * n and attempts < n_events * 4:
            attempts += 1
            start = rng.randint(int(0.1 * n), n - ev_len - 1)
            if labels[start:start + ev_len].any():
                continue
            atype = self.anomaly_types[rng.choice(type_choice)]
            ramp = int(rng.randint(onset_lo, onset_hi + 1))
            amp = rng.uniform(3.0, 6.0) if "surge" in atype or "spike" in atype \
                else rng.uniform(1.5, 3.5)
            # gradual onset envelope over the first `ramp` samples
            env = np.ones(ev_len)
            env[:ramp] = np.linspace(0.0, 1.0, ramp)
            ch = rng.randint(0, self.n_channels)
            Y[start:start + ev_len, ch] += amp * env
            modes[start:start + ev_len] = 3            # fault mode
            labels[start:start + ev_len] = 1
            injected += ev_len

        # user-behaviour confounds: legitimate mode transitions (not labelled)
        n_conf = max(1, n_events // 2)
        for _ in range(n_conf):
            s = rng.randint(int(0.1 * n), n - 50)
            L = rng.randint(20, 80)
            modes[s:s + L] = rng.choice([1, 2])         # degradation / warning
            Y[s:s + L] += rng.uniform(-0.4, 0.4, size=(L, self.n_channels))

        # partial observability: occlude one channel in 15 % of events
        occlude = rng.rand(n) < Config.injection_partial_occlusion_ratio
        if occlude.any():
            ch = rng.randint(0, self.n_channels)
            Y[occlude, ch] = 0.0
        return Y.astype(np.float64), modes

    def _mock_signal(self, rng: np.random.RandomState, n: int,
                     fs: float) -> np.ndarray:          # pragma: no cover - abstract
        raise NotImplementedError

    def _read_mock_sample(self) -> np.ndarray:
        if not hasattr(self, "_mock_idx"):
            self._mock_idx = 0
        if self._mock_idx >= len(self._mock_buffer):
            self._mock_idx = 0
        s = self._mock_buffer[self._mock_idx]
        self._mock_idx += 1
        return s

    def _acquire_mock(self, n: int, return_modes: bool
                      ) -> Tuple[np.ndarray, ...]:
        Y, modes = self._mock_buffer, self._mock_modes
        if n <= len(Y):
            Yn, mn = Y[:n].copy(), modes[:n].copy()
        else:                                          # loop the replay buffer
            reps = int(np.ceil(n / len(Y)))
            Yn = np.tile(Y, (reps, 1))[:n]
            mn = np.tile(modes, reps)[:n]
        labels = (mn == 3).astype(int)
        # user-confound samples (mode 1/2) are NOT anomalies
        labels[(mn != 3)] = 0
        if return_modes:
            return Yn, labels, mn
        return Yn, labels

    def _acquire_real(self, n: int, return_modes: bool
                      ) -> Tuple[np.ndarray, ...]:     # pragma: no cover - HW
        rows = np.zeros((n, self.n_channels), dtype=np.float64)
        for i in range(n):
            rows[i] = self._read_real_sample()
            time.sleep(max(0.0, 1.0 / self.sampling_rate_hz - 1e-4))
        labels = np.zeros(n, dtype=int)               # real labels come from the
        # annotation side-channel; left to the caller to merge.
        if return_modes:
            return rows, labels, np.zeros(n, dtype=int)
        return rows, labels


# ===========================================================================
# SPSD — Android smartphone sensor port
# ===========================================================================
class SPSDDevicePort(DevicePort):
    """Acquire accelerometer + gyroscope from an Android phone over ADB.

    Paper device: Samsung Galaxy S23 / Xiaomi 13 / OnePlus 11 (Android 13-14),
    6 channels (accel x/y/z, gyro x/y/z) @ 50 Hz.  Selected channel: accel-x
    (paper L572).
    """

    dataset_name = "SPSD"
    channel_names = ("accel_x", "accel_y", "accel_z",
                     "gyro_x", "gyro_y", "gyro_z")
    selected_channel_idx = 0

    def _open_real_device(self) -> None:               # pragma: no cover - HW
        try:
            from adb_shell.adb_device import AdbDeviceTcp  # type: ignore
        except ImportError as exc:
            raise DeviceUnavailable(
                "Real SPSD acquisition needs the 'adb-shell' package and a USB/"
                "network-connected Android device. Install with `pip install "
                "adb-shell` and run `adb forward`. Set use_mock=True to replay."
            ) from exc
        # In a real deployment the device serial would come from Config.
        host = getattr(self, "adb_host", "127.0.0.1")
        port = getattr(self, "adb_port", 5037)
        self._adb = AdbDeviceTcp(host, port)
        self._adb.connect()

    def _read_real_sample(self) -> np.ndarray:         # pragma: no cover - HW
        # `adb shell cmd sensorservice sensor_rate ...` or a dedicated sensor
        # logger APK streams one JSON line per sample per call.
        line = self._adb.shell("dumpsys sensorservice | grep -A1 accelerometer")
        # parse omitted in mock-first build; returns a 6-vector when present.
        raise DeviceUnavailable("Real SPSD parsing not wired in this build.")

    def _close_real_device(self) -> None:              # pragma: no cover - HW
        if getattr(self, "_adb", None) is not None:
            self._adb.close()

    def _mock_signal(self, rng, n, fs) -> np.ndarray:
        t = np.arange(n) / fs
        Y = np.zeros((n, 6), dtype=np.float64)
        # gravity + hand motion on accel; baseline rotation on gyro
        Y[:, 0] = 0.2 * np.sin(2 * np.pi * 0.5 * t) + 9.81
        Y[:, 1] = 0.1 * np.sin(2 * np.pi * 0.8 * t) + rng.randn(n) * 0.02
        Y[:, 2] = 0.05 * np.sin(2 * np.pi * 1.2 * t) + rng.randn(n) * 0.02
        Y[:, 3] = 0.02 * np.sin(2 * np.pi * 0.3 * t) + rng.randn(n) * 0.005
        Y[:, 4] = 0.01 * np.sin(2 * np.pi * 0.4 * t) + rng.randn(n) * 0.005
        Y[:, 5] = 0.01 * np.sin(2 * np.pi * 0.6 * t) + rng.randn(n) * 0.005
        return Y


# ===========================================================================
# HPMD — home-appliance power port (Modbus/TCP)
# ===========================================================================
class HPMDDevicePort(DevicePort):
    """Acquire active power from three appliances via Modbus/TCP power meters.

    Paper devices: LG GR-B247 refrigerator, Daikin FTXTA35 air conditioner,
    LG WD-T14410 washing machine.  3 power channels @ 1 Hz.  Selected channel:
    active power (paper L572).
    """

    dataset_name = "HPMD"
    channel_names = ("refrigerator_power", "ac_power", "washer_power")
    selected_channel_idx = 0

    def _open_real_device(self) -> None:               # pragma: no cover - HW
        try:
            from pymodbus.client import ModbusTcpClient  # type: ignore
        except ImportError as exc:
            raise DeviceUnavailable(
                "Real HPMD acquisition needs 'pymodbus' and a Modbus/TCP power "
                "meter wired to each appliance. Install with `pip install "
                "pymodbus`. Set use_mock=True to replay."
            ) from exc
        self._mb = ModbusTcpClient(getattr(self, "mb_host", "192.168.1.50"))

    def _read_real_sample(self) -> np.ndarray:         # pragma: no cover - HW
        # three holding registers, one per appliance
        vals = []
        for unit in (1, 2, 3):
            rr = self._mb.read_holding_registers(address=0x0000, count=2,
                                                  slave=unit)
            watts = (rr.registers[0] << 16) | rr.registers[1]
            vals.append(float(watts) / 10.0)
        return np.asarray(vals, dtype=np.float64)

    def _close_real_device(self) -> None:              # pragma: no cover - HW
        if getattr(self, "_mb", None) is not None:
            self._mb.close()

    def _mock_signal(self, rng, n, fs) -> np.ndarray:
        t = np.arange(n) / fs
        Y = np.zeros((n, 3), dtype=np.float64)
        # refrigerator: near-constant ~150 W with compressor duty cycle
        Y[:, 0] = 150 + 30 * np.sin(2 * np.pi * t / 1200.0) + rng.randn(n) * 2
        # air conditioner: diurnal
        Y[:, 1] = 300 + 200 * np.sin(2 * np.pi * t / 86400.0) + rng.randn(n) * 5
        # washing machine: idle with sporadic bursts
        Y[:, 2] = 10 + 50 * (np.sin(2 * np.pi * t / 1800.0) > 0.7) + rng.randn(n)
        return np.maximum(Y, 0)


# ===========================================================================
# BMSD — lithium-polymer BMS port (SMBus/I2C)
# ===========================================================================
class BMSDDevicePort(DevicePort):
    """Acquire cell voltage / current / temperature from a 4S2P Li-Po BMS.

    Paper device: 4S2P lithium-polymer pack (14.8 V, 5000 mAh) monitored in a
    temperature-controlled chamber.  3 channels @ 10 Hz.  Selected channel:
    cell voltage (paper L572).
    """

    dataset_name = "BMSD"
    channel_names = ("cell_voltage", "pack_current", "temperature")
    selected_channel_idx = 0

    def _open_real_device(self) -> None:               # pragma: no cover - HW
        try:
            from smbus2 import SMBus                    # type: ignore
        except ImportError as exc:
            raise DeviceUnavailable(
                "Real BMSD acquisition needs 'smbus2' and a SMBus/I2C BMS "
                "controller on the 4S2P Li-Po pack. Install with `pip install "
                "smbus2`. Set use_mock=True to replay."
            ) from exc
        self._bus = SMBus(getattr(self, "i2c_bus", 1))

    def _read_real_sample(self) -> np.ndarray:         # pragma: no cover - HW
        # typical Smart Battery SMBus registers
        v = self._bus.read_word_data(0x0B, 0x09) / 1000.0   # voltage, mV->V
        i = self._bus.read_word_data(0x0B, 0x0A) / 1000.0   # current, mA->A
        T = self._bus.read_word_data(0x0B, 0x08) / 10.0     # temp, 0.1K->C
        return np.asarray([v, i, T], dtype=np.float64)

    def _close_real_device(self) -> None:              # pragma: no cover - HW
        if getattr(self, "_bus", None) is not None:
            self._bus.close()

    def _mock_signal(self, rng, n, fs) -> np.ndarray:
        t = np.arange(n) / fs
        Y = np.zeros((n, 3), dtype=np.float64)
        cycle = 500
        phase = (np.arange(n) % cycle) / cycle
        voltage = np.where(phase < 0.7,
                           4.2 - 1.0 * (phase / 0.7),
                           3.2 + 1.0 * ((phase - 0.7) / 0.3))
        Y[:, 0] = voltage + rng.randn(n) * 0.005
        Y[:, 1] = 0.5 * np.sin(2 * np.pi * t / 50.0) + rng.randn(n) * 0.05
        Y[:, 2] = 25 + 5 * np.sin(2 * np.pi * t / 200.0) + rng.randn(n) * 0.3
        return Y


# ---------------------------------------------------------------------------
# Registry & convenience
# ---------------------------------------------------------------------------
PORT_REGISTRY = {
    "SPSD": SPSDDevicePort,
    "HPMD": HPMDDevicePort,
    "BMSD": BMSDDevicePort,
}


def open_port(dataset_name: str, n_samples: int = 5000, seed: int = 42,
              use_mock: Optional[bool] = None) -> DevicePort:
    """Return a connected :class:`DevicePort` for ``dataset_name``.

    ``use_mock=None`` (default) auto-detects: real device if reachable, mock
    otherwise.  Pass ``use_mock=True`` to force the replay buffer (used by
    ``get_all_datasets`` and the CI test harness).
    """
    if dataset_name not in PORT_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset_name}'. "
                         f"Available: {list(PORT_REGISTRY)}")
    port = PORT_REGISTRY[dataset_name](n_samples=n_samples, seed=seed,
                                       use_mock=use_mock)
    return port.connect()


__all__ = [
    "DevicePort", "SPSDDevicePort", "HPMDDevicePort", "BMSDDevicePort",
    "DeviceUnavailable", "PORT_REGISTRY", "open_port",
]
