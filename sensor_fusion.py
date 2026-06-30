"""
===============================================================================
Sensor Fusion Engine
===============================================================================

Extracts physiological and motion features from incoming sensor data.

Responsibilities:
    - Maintain rolling sensor buffers
    - Compute statistical features
    - Produce feature vector for Exercise Detection

Author : Debjyoti Saha
Version : 1.0
===============================================================================
"""

from collections import deque
from dataclasses import dataclass
import numpy as np

from exercise_config import DEFAULT_EXERCISE_CONFIG

# =============================================================================
# Feature Vector
# =============================================================================


@dataclass
class FeatureVector:

    # ---------- CGM ----------
    glucose: float
    glucose_slope: float
    glucose_acceleration: float
    glucose_std: float

    # ---------- Heart ----------
    heart_rate: float
    heart_rate_change: float

    # ---------- Accelerometer ----------
    accel_rms: float
    accel_std: float

    # ---------- Gyroscope ----------
    gyro_rms: float
    gyro_std: float

    # ---------- Activity ----------
    step_frequency: float

    # ---------- Therapy ----------
    insulin_on_board: float

    # ---------- Metadata ----------
    sample_count: int


# =============================================================================
# Sensor Fusion Engine
# =============================================================================


class SensorFusionEngine:

    def __init__(self, config=DEFAULT_EXERCISE_CONFIG):

        self.config = config

        window_size = int(config.window_duration_sec / config.sampling_interval_sec)

        # Rolling buffers

        self.glucose = deque(maxlen=window_size)

        self.heart_rate = deque(maxlen=window_size)

        self.acc_x = deque(maxlen=window_size)
        self.acc_y = deque(maxlen=window_size)
        self.acc_z = deque(maxlen=window_size)

        self.gyro_x = deque(maxlen=window_size)
        self.gyro_y = deque(maxlen=window_size)
        self.gyro_z = deque(maxlen=window_size)

        self.step_count = deque(maxlen=window_size)

        self.iob = deque(maxlen=window_size)

    # =========================================================================

    def update(self, payload: dict):
        """
        Add one MQTT payload into rolling buffers.
        """

        self.glucose.append(float(payload.get("glucose", np.nan)))

        self.heart_rate.append(float(payload.get("heart_rate", np.nan)))

        self.acc_x.append(float(payload.get("acc_x", np.nan)))
        self.acc_y.append(float(payload.get("acc_y", np.nan)))
        self.acc_z.append(float(payload.get("acc_z", np.nan)))

        self.gyro_x.append(float(payload.get("gyro_x", np.nan)))
        self.gyro_y.append(float(payload.get("gyro_y", np.nan)))
        self.gyro_z.append(float(payload.get("gyro_z", np.nan)))

        self.step_count.append(float(payload.get("step_count", np.nan)))

        self.iob.append(float(payload.get("iob", np.nan)))

    # =========================================================================

    @staticmethod
    def compute_rms(x, y, z):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)

        mag_sq = x**2 + y**2 + z**2

        return float(np.sqrt(np.nanmean(mag_sq)))

    # =========================================================================

    @staticmethod
    def compute_std(values):

        return np.nanstd(values)

    # =========================================================================

    @staticmethod
    def compute_slope(values):
        """
        Compute signal slope using linear least-squares regression.
        More robust than simple finite differences.
        """

        values = np.asarray(values, dtype=float)

        # Remove NaNs
        mask = np.isfinite(values)
        values = values[mask]

        if len(values) < 2:
            return 0.0

        # Time axis (sample index)
        x = np.arange(len(values))

        # y = mx + c
        slope, _ = np.polyfit(x, values, 1)

        return float(slope)

    # =========================================================================

    @staticmethod
    def compute_acceleration(values):
        """
        Compute second derivative (change in slope)
        using consecutive least-squares slopes.
        """

        values = np.asarray(values, dtype=float)

        mask = np.isfinite(values)
        values = values[mask]

        if len(values) < 5:
            return 0.0

        mid = len(values) // 2

        x1 = np.arange(mid)
        x2 = np.arange(len(values) - mid)

        slope1, _ = np.polyfit(x1, values[:mid], 1)
        slope2, _ = np.polyfit(x2, values[mid:], 1)

        return float(slope2 - slope1)

    # =========================================================================

    def compute_step_frequency(self):

        if len(self.step_count) < 2:
            return 0.0

        steps = self.step_count[-1] - self.step_count[0]

        elapsed = len(self.step_count) * self.config.sampling_interval_sec

        if elapsed <= 0:
            return 0.0

        return steps / elapsed

    # =========================================================================

    def get_feature_vector(self):
        """
        Compute all features.

        Returns
        -------
        FeatureVector
        """

        accel_rms = self.compute_rms(self.acc_x, self.acc_y, self.acc_z)

        gyro_rms = self.compute_rms(self.gyro_x, self.gyro_y, self.gyro_z)

        return FeatureVector(
            glucose=self.glucose[-1] if self.glucose else np.nan,
            glucose_slope=self.compute_slope(self.glucose),
            glucose_acceleration=self.compute_acceleration(self.glucose),
            glucose_std=self.compute_std(self.glucose),
            heart_rate=self.heart_rate[-1] if self.heart_rate else np.nan,
            heart_rate_change=self.compute_slope(self.heart_rate),
            accel_rms=accel_rms,
            accel_std=self.compute_std(
                np.sqrt(
                    np.array(self.acc_x) ** 2
                    + np.array(self.acc_y) ** 2
                    + np.array(self.acc_z) ** 2
                )
            ),
            gyro_rms=gyro_rms,
            gyro_std=self.compute_std(
                np.sqrt(
                    np.array(self.gyro_x) ** 2
                    + np.array(self.gyro_y) ** 2
                    + np.array(self.gyro_z) ** 2
                )
            ),
            step_frequency=self.compute_step_frequency(),
            insulin_on_board=self.iob[-1] if self.iob else np.nan,
            sample_count=len(self.glucose),
        )
