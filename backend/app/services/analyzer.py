import numpy as np
import pandas as pd
from typing import Dict, Any
from datetime import datetime
import logging

from .model_loader import get_model_loader
from .predictor import Predictor
from .validation import InputValidator

logger = logging.getLogger(__name__)


class CompleteAnalyzer:
    """Production-grade complete analyzer with all 4 agents"""

    def __init__(self):
        self.model_loader = get_model_loader()
        self.predictor = Predictor(self.model_loader)
        self.validator = InputValidator(
            self.model_loader.get_feature_names() or []
        )
        self.analysis_count = 0

    def analyze_patient(
        self,
        patient_features: Dict[str, float],
        patient_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Complete end-to-end analysis with all 4 agents

        Includes:
        - Input validation
        - Feature preparation & scaling
        - Disease classification
        - Trend forecasting
        - Risk assessment
        - Result compilation
        """

        self.analysis_count += 1
        analysis_id = f"ANALYSIS_{self.analysis_count}_{datetime.now().timestamp()}"

        logger.info(f"Starting analysis {analysis_id}...")
        logger.info(f"Patient info: {patient_info}")

        try:
            start_time = datetime.now()

            # Convert Pydantic model to dict if needed
            if (
                not isinstance(patient_features, dict)
                and hasattr(patient_features, "model_dump")
            ):
                patient_features = patient_features.model_dump()

            # Step 1: Validate input
            is_valid, validation_result = self.validator.validate_lab_values(
                patient_features
            )

            if not is_valid:
                logger.warning(f"Validation warnings/errors: {validation_result}")

            # Step 2: Prepare features
            features_array = self._prepare_features(patient_features)

            if features_array is None:
                logger.error("Feature preparation failed")

                return {
                    "analysis_id": analysis_id,
                    "status": "failed",
                    "error": "Feature preparation failed",
                    "timestamp": datetime.now().isoformat()
                }

            # Step 3: Agent 2 - Disease Classification
            logger.info("Running Agent 2: Disease Classification...")
            disease_result = self.predictor.predict_disease(features_array)

            # Step 4: Agent 3 - Trend Forecasting
            logger.info("Running Agent 3: Trend Forecasting...")
            trend_result = self.predictor.forecast_trends(patient_features)

            # Step 5: Agent 4 - Risk Assessment
            logger.info("Running Agent 4: Risk Assessment...")
            risk_result = self.predictor.assess_risk(features_array)

            # Step 6: Compile results
            complete_analysis = {
                "analysis_id": analysis_id,
                "timestamp": datetime.now().isoformat(),
                "patient_info": patient_info or {},

                "agent1_parsing": {
                    "status": "completed",
                    "parameters_extracted": len(patient_features),
                    "parameters_filled": len([
                        f for f in (
                            self.model_loader.get_feature_names() or []
                        )
                        if f in patient_features
                    ])
                },

                "agent2_disease_detection": disease_result,
                "agent3_trend_forecasting": trend_result,
                "agent4_risk_assessment": risk_result,

                "summary": self._generate_summary(
                    disease_result,
                    trend_result,
                    risk_result
                ),

                "validation_warnings": validation_result.get(
                    "warnings",
                    []
                ),

                "processing_time_ms": round(
                    (datetime.now() - start_time).total_seconds() * 1000,
                    2
                ),

                "status": "success"
            }

            logger.info(f"Analysis {analysis_id} completed successfully")

            return complete_analysis

        except Exception as e:
            logger.error(
                f"Analysis {analysis_id} failed: {e}",
                exc_info=True
            )

            return {
                "analysis_id": analysis_id,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _prepare_features(
        self,
        patient_features: Dict[str, float]
    ) -> np.ndarray:
        """Prepare and scale features safely"""

        try:
            feature_names = self.model_loader.get_feature_names()
            scaler = self.model_loader.get_scaler()

            if feature_names is None:
                logger.error("Feature names not available")
                return None

            # Create feature dictionary
            feature_dict = {}

            for fname in feature_names:

                value = patient_features.get(fname, 0.0)

                try:
                    value = float(value)
                except:
                    value = 0.0

                feature_dict[fname] = value

            # Create dataframe with exact feature order
            X = pd.DataFrame([feature_dict])

            X = X[feature_names]

            logger.info(f"Feature dataframe shape: {X.shape}")

            # Handle NaN / Inf
            X = X.replace([np.inf, -np.inf], 0)
            X = X.fillna(0)

            # Scale features
            if scaler is not None:
                X_scaled = scaler.transform(X)
                return X_scaled[0]

            return X.values[0]

        except Exception as e:
            logger.error(
                f"Error preparing features: {e}",
                exc_info=True
            )
            return None

    def _generate_summary(
        self,
        disease_result: Dict,
        trend_result: Dict,
        risk_result: Dict
    ) -> str:
        """Generate human-readable summary"""

        summary = "=" * 70 + "\n"
        summary += "CLINICAL ANALYSIS SUMMARY\n"
        summary += "=" * 70 + "\n\n"

        # Disease Section
        if (
            "disease_type" in disease_result
            and "error" not in disease_result
        ):
            summary += "CONDITION DETECTED:\n"
            summary += f"  • Type: {disease_result['disease_type']}\n"
            summary += (
                f"  • Confidence: "
                f"{disease_result['confidence'] * 100:.1f}%\n"
            )
            summary += "\n"

        # Trend Section
        if trend_result and "error" not in trend_result:

            summary += "PARAMETER TRENDS (Next 90 Days):\n"

            increasing = []
            decreasing = []

            for param, forecast in trend_result.items():

                if "trend" in forecast:

                    if forecast["trend"] == "increasing":
                        increasing.append(f"{param.upper()} ↑")

                    elif forecast["trend"] == "decreasing":
                        decreasing.append(f"{param.upper()} ↓")

            if increasing:
                summary += (
                    f"  Increasing: {', '.join(increasing)}\n"
                )

            if decreasing:
                summary += (
                    f"  Decreasing: {', '.join(decreasing)}\n"
                )

            summary += "\n"

        # Risk Section
        if (
            "risk_level" in risk_result
            and "error" not in risk_result
        ):

            summary += "OVERALL HEALTH RISK:\n"

            summary += (
                f"  • Risk Level: "
                f"{risk_result['risk_level'].upper()}\n"
            )

            summary += (
                f"  • Risk Score: "
                f"{risk_result['risk_score']}/100\n"
            )

            summary += (
                f"  • Confidence: "
                f"{risk_result['confidence'] * 100:.1f}%\n"
            )

            if "recommendations" in risk_result:

                summary += "\nRECOMMENDATIONS:\n"

                for i, rec in enumerate(
                    risk_result["recommendations"],
                    1
                ):
                    summary += f"  {i}. {rec}\n"

        summary += "\n" + "=" * 70

        return summary


# Global analyzer instance
_analyzer = None


def get_analyzer() -> CompleteAnalyzer:
    """Get or create analyzer"""

    global _analyzer

    if _analyzer is None:
        _analyzer = CompleteAnalyzer()

    return _analyzer