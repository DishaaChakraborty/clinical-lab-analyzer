import pandas as pd
import re
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class Agent1Parser:
    """
    Agent 1: FILE PARSING
    Parse CSV/TXT lab reports and extract parameters
    """
    
    PARAMETER_PATTERNS = {
        'glucose': r'glucose[:\s]+(\d+\.?\d*)',
        'cholesterol': r'(?:total\s+)?cholesterol[:\s]+(\d+\.?\d*)',
        'hdl': r'hdl[:\s]+(\d+\.?\d*)',
        'ldl': r'ldl[:\s]+(\d+\.?\d*)',
        'triglycerides': r'triglyceride[s]?[:\s]+(\d+\.?\d*)',
        'hemoglobin': r'hemoglobin[:\s]+(\d+\.?\d*)',
        'creatinine': r'creatinine[:\s]+(\d+\.?\d*)',
        'bmi': r'bmi[:\s]+(\d+\.?\d*)',
        'systolic_bp': r'(?:systolic|bp[\s-]?sys)[:\s]+(\d+\.?\d*)',
        'diastolic_bp': r'(?:diastolic|bp[\s-]?dia)[:\s]+(\d+\.?\d*)',
        'uric_acid': r'uric\s+acid[:\s]+(\d+\.?\d*)',
        'wbc': r'wbc[:\s]+(\d+\.?\d*)',
        'potassium': r'potassium[:\s]+(\d+\.?\d*)',
    }
    
    def __init__(self):
        self.parsed_data = []
    
    def parse_csv(self, filepath: str) -> pd.DataFrame:
        """Parse CSV file"""
        print(f"Parsing CSV: {filepath}")
        df = pd.read_csv(filepath)
        return df
    
    def parse_text(self, text: str) -> Dict:
        """Parse text format report"""
        parsed = {}
        
        for param_name, pattern in self.PARAMETER_PATTERNS.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                parsed[param_name] = float(match.group(1))
        
        return parsed
    
    def validate_parameters(self, params: Dict) -> bool:
        """Validate extracted parameters"""
        required = ['glucose', 'cholesterol', 'hemoglobin']
        return all(p in params for p in required)
    
    def parse_report(self, filepath: str) -> Dict:
        """
        Parse any report file (CSV or TXT)
        Returns: Extracted parameters dictionary
        """
        
        if filepath.endswith('.csv'):
            df = self.parse_csv(filepath)
            return df.to_dict('records')[0] if len(df) > 0 else {}
        
        elif filepath.endswith('.txt'):
            with open(filepath, 'r') as f:
                text = f.read()
            return self.parse_text(text)
        
        else:
            raise ValueError(f"Unsupported file format: {filepath}")


if __name__ == "__main__":
    parser = Agent1Parser()
    # Test with synthetic data
    df = parser.parse_csv('backend/ml/data/synthetic_data.csv')
    print(f"✓ Parsed {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")