import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    def __init__(self, num_patients: int = 1000, seed: int = 42):
        self.num_patients = num_patients
        self.data = []
        random.seed(seed)
        np.random.seed(seed)

    # =============================
    # HEALTHY (STABLE TREND) - Now with overlap
    # =============================
    def generate_healthy_patient(self, patient_id: int) -> List[Dict]:
        records = []

        num_reports = np.random.randint(6, 12)

        # Overlap with other classes - healthy can have high values too
        base = {
            'glucose': np.random.uniform(85, 130),  # Wider range
            'cholesterol': np.random.uniform(150, 220),
            'hdl': np.random.uniform(35, 70),  # Lower min - overlaps with metabolic
            'ldl': np.random.uniform(70, 150),
            'triglycerides': np.random.uniform(70, 180),  # Higher max
            'hemoglobin': np.random.uniform(12, 17),  # Overlaps with anemia
            'creatinine': np.random.uniform(0.6, 1.3),  # Overlaps with kidney
        }

        for report_num in range(1, num_reports + 1):
            report_date = (datetime.now() - timedelta(days=30 * (num_reports - report_num))).isoformat()

            record = {
                'patient_id': patient_id,
                'age': np.random.randint(20, 70),  # Wider age range
                'gender': np.random.choice(['M', 'F']),
                'report_number': report_num,
                'report_date': report_date,

                'glucose': base['glucose'] + np.random.normal(0, 8),  # More variance
                'cholesterol': base['cholesterol'] + np.random.normal(0, 15),
                'hdl': base['hdl'] + np.random.normal(0, 5),
                'ldl': base['ldl'] + np.random.normal(0, 10),
                'triglycerides': base['triglycerides'] + np.random.normal(0, 15),
                'hemoglobin': base['hemoglobin'] + np.random.normal(0, 1),
                'creatinine': base['creatinine'] + np.random.normal(0, 0.1),

                'bmi': np.random.uniform(18, 32),  # Wider BMI
                'systolic_bp': np.random.uniform(100, 150),  # Overlaps
                'diastolic_bp': np.random.uniform(65, 95),
                'uric_acid': np.random.uniform(3.0, 7.0),
                'wbc': np.random.uniform(4.0, 11.0),
                'potassium': np.random.uniform(3.0, 5.5),

                # Label based on ACTUAL health indicators (with noise)
                'risk_label': self._calculate_risk_label(
                    glucose=base['glucose'],
                    hemoglobin=base['hemoglobin'],
                    creatinine=base['creatinine'],
                    hdl=base['hdl'],
                    systolic_bp=base.get('systolic_bp', 120)
                )
            }
            records.append(record)

        return records

    def _calculate_risk_label(self, glucose, hemoglobin, creatinine, hdl, systolic_bp):
        """
        Calculate risk based on multiple factors - CLEAR LEARNABLE patterns
        No noise added - clean signals for ML to learn
        """
        risk_score = 0
        
        # Clear diagnostic signals (no noise)
        if glucose > 120: risk_score += 3
        elif glucose > 100: risk_score += 1
        
        if hemoglobin < 11.5: risk_score += 3
        elif hemoglobin < 13: risk_score += 1
        
        if creatinine > 1.1: risk_score += 3
        elif creatinine > 0.9: risk_score += 1
        
        if hdl < 35: risk_score += 2
        elif hdl < 45: risk_score += 1
        
        if systolic_bp > 140: risk_score += 3
        elif systolic_bp > 120: risk_score += 1
        
        # Map to classes (deterministic)
        if risk_score <= 2:
            return 'Low'
        elif risk_score <= 5:
            return 'Medium'
        else:
            return 'High'

    # =============================
    # METABOLIC (STRONG INCREASING TREND) - With overlap
    # =============================
    def generate_metabolic_syndrome_patient(self, patient_id: int) -> List[Dict]:
        records = []
        num_reports = np.random.randint(6, 12)

        # Overlap with healthy - not all metabolic have extreme values
        slopes = {
            'glucose': np.random.uniform(2, 8),  # Smaller slope
            'cholesterol': np.random.uniform(3, 8),
            'triglycerides': np.random.uniform(4, 10),
        }

        base = {
            'glucose': np.random.uniform(90, 130),  # Can start normal
            'cholesterol': np.random.uniform(160, 220),
            'hdl': np.random.uniform(30, 55),  # Overlaps with healthy
            'ldl': np.random.uniform(90, 160),
            'triglycerides': np.random.uniform(100, 180),
        }

        for report_num in range(1, num_reports + 1):
            report_date = (datetime.now() - timedelta(days=30 * (num_reports - report_num))).isoformat()

            glucose = base['glucose'] + report_num * slopes['glucose'] + np.random.normal(0, 5)
            cholesterol = base['cholesterol'] + report_num * slopes['cholesterol'] + np.random.normal(0, 8)
            triglycerides = base['triglycerides'] + report_num * slopes['triglycerides'] + np.random.normal(0, 8)

            record = {
                'patient_id': patient_id,
                'age': np.random.randint(25, 70),  # Wider
                'gender': np.random.choice(['M', 'F']),
                'report_number': report_num,
                'report_date': report_date,

                'glucose': glucose,
                'cholesterol': cholesterol,
                'hdl': base['hdl'] - report_num * 0.3 + np.random.normal(0, 3),
                'ldl': base['ldl'] + report_num * 2 + np.random.normal(0, 5),
                'triglycerides': triglycerides,
                'hemoglobin': np.random.uniform(12, 16),  # Overlaps
                'creatinine': np.random.uniform(0.7, 1.4),  # Overlaps

                'bmi': np.random.uniform(22, 35),
                'systolic_bp': np.random.uniform(110, 160),  # Overlaps
                'diastolic_bp': np.random.uniform(70, 100),
                'uric_acid': np.random.uniform(4.0, 7.5),
                'wbc': np.random.uniform(5.0, 10.0),
                'potassium': np.random.uniform(3.2, 5.5),

                # Use same risk calculation
                'risk_label': self._calculate_risk_label(
                    glucose=glucose,
                    hemoglobin=np.random.uniform(12, 16),
                    creatinine=np.random.uniform(0.7, 1.4),
                    hdl=base['hdl'] - report_num * 0.3,
                    systolic_bp=np.random.uniform(110, 160)
                )
            }
            records.append(record)

        return records

    # =============================
    # ANEMIA (DECLINING HEMOGLOBIN) - With overlap
    # =============================
    def generate_anemia_patient(self, patient_id: int) -> List[Dict]:
        records = []
        num_reports = np.random.randint(6, 12)

        for report_num in range(1, num_reports + 1):
            report_date = (datetime.now() - timedelta(days=30 * (num_reports - report_num))).isoformat()

            # Not always declining - some stay stable
            if np.random.random() < 0.3:
                hemoglobin = np.random.uniform(11, 15)  # Stable
            else:
                hemoglobin = np.random.uniform(12, 15) - report_num * np.random.uniform(0.2, 0.6)

            record = {
                'patient_id': patient_id,
                'age': np.random.randint(20, 70),  # Wider
                'gender': np.random.choice(['F', 'M', 'F']),  # More realistic ratio
                'report_number': report_num,
                'report_date': report_date,

                'glucose': np.random.uniform(85, 130),  # Overlaps
                'cholesterol': np.random.uniform(150, 220),
                'hdl': np.random.uniform(35, 65),
                'ldl': np.random.uniform(80, 160),
                'triglycerides': np.random.uniform(80, 180),
                'hemoglobin': hemoglobin,
                'creatinine': np.random.uniform(0.6, 1.4),  # Overlaps

                'bmi': np.random.uniform(18, 32),
                'systolic_bp': np.random.uniform(100, 150),
                'diastolic_bp': np.random.uniform(65, 95),
                'uric_acid': np.random.uniform(3.5, 7.0),
                'wbc': np.random.uniform(4.5, 10.0),
                'potassium': np.random.uniform(3.2, 5.5),

                # Use same risk calculation
                'risk_label': self._calculate_risk_label(
                    glucose=np.random.uniform(85, 130),
                    hemoglobin=hemoglobin,
                    creatinine=np.random.uniform(0.6, 1.4),
                    hdl=np.random.uniform(35, 65),
                    systolic_bp=np.random.uniform(100, 150)
                )
            }
            records.append(record)

        return records

    # =============================
    # MAIN GENERATOR
    # =============================
    def generate_all_patients(self) -> pd.DataFrame:
        print("\nGenerating dataset...\n")

        patient_id = 1

        for _ in range(300):
            self.data.extend(self.generate_healthy_patient(patient_id))
            patient_id += 1

        for _ in range(300):
            self.data.extend(self.generate_metabolic_syndrome_patient(patient_id))
            patient_id += 1

        for _ in range(200):
            self.data.extend(self.generate_anemia_patient(patient_id))
            patient_id += 1

        df = pd.DataFrame(self.data)

        print(f"\n✓ Generated {len(df)} records\n")
        print(df['risk_label'].value_counts())

        return df

    def save_to_csv(self, filepath='ml/data/synthetic_data.csv'):
        df = pd.DataFrame(self.data)
        df.to_csv(filepath, index=False)
        print(f"✓ Saved to {filepath}")


def generate_training_data():
    generator = SyntheticDataGenerator()
    df = generator.generate_all_patients()
    generator.save_to_csv()
    return df


if __name__ == "__main__":
    generate_training_data()