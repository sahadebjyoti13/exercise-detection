"""
===============================================================================
Exercise Detection Configuration
===============================================================================

Central configuration for exercise detection and sensor fusion.

Author : Debjyoti Saha
Version : 1.0
===============================================================================
"""

from dataclasses import dataclass


@dataclass
class ExerciseConfig:

    # ---------------------------------------------------------
    # Sampling
    # ---------------------------------------------------------
    sampling_interval_sec: float = 5.0

    # ---------------------------------------------------------
    # Rolling Window
    # ---------------------------------------------------------
    window_duration_sec: int = 60

    # ---------------------------------------------------------
    # Sensor Enable Flags
    # ---------------------------------------------------------
    use_cgm: bool = True

    use_accelerometer: bool = True

    use_gyroscope: bool = True

    use_heart_rate: bool = True

    use_step_counter: bool = True

    use_iob: bool = True

    # ---------------------------------------------------------
    # Activity Thresholds
    # ---------------------------------------------------------
    resting_hr: float = 75.0

    exercise_hr: float = 105.0

    vigorous_hr: float = 140.0

    accel_rms_threshold: float = 0.40

    gyro_rms_threshold: float = 0.15

    glucose_drop_threshold: float = -1.2

    step_frequency_threshold: float = 1.2

    # ---------------------------------------------------------
    # Confidence
    # ---------------------------------------------------------
    confidence_threshold: float = 0.70

    # ---------------------------------------------------------
    # Machine Learning
    # ---------------------------------------------------------
    model_path: str = "models/exercise_rf.pkl"

    use_ml_model: bool = False

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------
    publish_confidence: bool = True

    publish_activity_type: bool = True

    publish_intensity: bool = True


DEFAULT_EXERCISE_CONFIG = ExerciseConfig()
