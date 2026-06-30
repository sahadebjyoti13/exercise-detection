"""
===============================================================================
Exercise Detection Algorithm
===============================================================================

Combines

    Sensor Fusion
    +
    Machine Learning
    +
    Rule-based Safety Logic

Author : Debjyoti Saha
Version : 1.0
===============================================================================
"""

from dataclasses import dataclass

from exercise_config import DEFAULT_EXERCISE_CONFIG
from sensor_fusion import SensorFusionEngine
from activity_models import ActivityRecognitionModel

# =============================================================================
# Output Structure
# =============================================================================


@dataclass
class ExerciseDecision:

    detected: bool

    confidence: float

    probability: float

    activity: str

    intensity: str

    met: float

    suggested_basal_factor: float

    reason: str


# =============================================================================
# Exercise Detector
# =============================================================================


class ExerciseDetector:

    def __init__(self, config=DEFAULT_EXERCISE_CONFIG):

        self.config = config

        self.sensor_fusion = SensorFusionEngine(config)

        self.ml_model = ActivityRecognitionModel(config)

    # =========================================================================

    def update(self, payload):
        """
        Process one MQTT payload.
        """

        self.sensor_fusion.update(payload)

        features = self.sensor_fusion.get_feature_vector()

        prediction = self.ml_model.predict(features)

        return self.make_decision(features, prediction)

    # =========================================================================

    def make_decision(self, features, prediction):
        """
        Final exercise decision.

        ML prediction alone is never trusted.

        Simple physiological safety checks are also applied.
        """

        detected = prediction.exercise

        confidence = prediction.confidence

        probability = prediction.probability

        activity = prediction.activity

        intensity = prediction.intensity

        met = prediction.met

        reason = []

        # ----------------------------------------------------------
        # Confidence check
        # ----------------------------------------------------------

        if confidence < self.config.confidence_threshold:

            detected = False

            reason.append("Low ML confidence")

        # ----------------------------------------------------------
        # Heart-rate sanity check
        # ----------------------------------------------------------

        if self.config.use_heart_rate:

            if features.heart_rate < self.config.resting_hr:

                detected = False

                reason.append("Heart rate below exercise threshold")

        # ----------------------------------------------------------
        # Accelerometer sanity check
        # ----------------------------------------------------------

        if self.config.use_accelerometer:

            if features.accel_rms < self.config.accel_rms_threshold:

                detected = False

                reason.append("Insufficient body movement")

        # ----------------------------------------------------------
        # Basal recommendation
        # ----------------------------------------------------------

        if len(reason) == 0:

            reason.append("Exercise detected")

        return ExerciseDecision(
            detected=detected,
            confidence=confidence,
            probability=probability,
            activity=activity,
            intensity=intensity,
            met=met,
            reason=", ".join(reason),
        )
