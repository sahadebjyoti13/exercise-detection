"""
===============================================================================
Activity Recognition Models
===============================================================================

Loads machine learning models for exercise recognition.

Responsibilities
----------------
- Load trained ML model
- Perform inference
- Return prediction probabilities

Author : Debjyoti Saha
Version : 1.0
===============================================================================
"""

from pathlib import Path
from typing import Dict

import joblib
import numpy as np

from exercise_config import DEFAULT_EXERCISE_CONFIG
from sensor_fusion import FeatureVector

# =============================================================================
# Activity Model
# =============================================================================


class ActivityRecognitionModel:

    def __init__(self, config=DEFAULT_EXERCISE_CONFIG):

        self.config = config

        self.model = None

        self.loaded = False

        self.load_model()

    # =========================================================================

    def load_model(self):
        """
        Load trained ML model.
        """

        model_path = Path(self.config.model_path)

        if not model_path.exists():

            print(f"[Exercise ML] Model not found : {model_path}")

            self.loaded = False

            return

        self.model = joblib.load(model_path)

        self.loaded = True

        print("[Exercise ML] Model loaded successfully.")

    # =========================================================================

    @staticmethod
    def feature_vector_to_numpy(features: FeatureVector):
        """
        Convert FeatureVector into ML input vector.
        """

        return np.array(
            [
                features.glucose,
                features.glucose_slope,
                features.glucose_acceleration,
                features.glucose_std,
                features.heart_rate,
                features.heart_rate_change,
                features.accel_rms,
                features.accel_std,
                features.gyro_rms,
                features.gyro_std,
                features.step_frequency,
                features.insulin_on_board,
            ]
        ).reshape(1, -1)

    # =========================================================================

    def predict(self, features: FeatureVector) -> Dict:
        """
        Run ML inference.
        """

        if not self.loaded:

            return {
                "exercise": False,
                "probability": 0.0,
                "confidence": 0.0,
                "activity": "Unknown",
            }

        x = self.feature_vector_to_numpy(features)

        prediction = self.model.predict(x)[0]

        probability = self.model.predict_proba(x)[0]

        confidence = float(np.max(probability))

        # ---------------------------------------------------------------------

        # Default mapping (can later be replaced with multiclass models)
        # ---------------------------------------------------------------------

        if prediction == 0:

            activity = "Rest"
            intensity = "None"
            met = 1.0

        else:

            activity = "Exercise"
            intensity = "Moderate"
            met = 4.5

        return {
            "exercise": bool(prediction),
            "probability": float(probability[1]),
            "confidence": confidence,
            "activity": activity,
            "intensity": intensity,
            "met": met,
        }
