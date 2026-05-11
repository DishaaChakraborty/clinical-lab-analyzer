import requests
import json

BASE_URL = "http://localhost:8000/api/predictions"

# Sample lab values
sample_lab_values = {
    "age": 55,
    "glucose": 150,
    "cholesterol": 220,
    "hdl": 35,
    "ldl": 160,
    "triglycerides": 180,
    "hemoglobin": 13.2,
    "creatinine": 1.1,
    "bmi": 28,
    "systolic_bp": 135,
    "diastolic_bp": 85,
    "uric_acid": 7.5,
    "wbc": 7.2,
    "potassium": 4.2,
    "gender": "M",
    # ... add all 50 features
}

def test_complete_analysis():
    """Test complete end-to-end analysis"""
    
    print("Testing COMPLETE ANALYSIS...")
    print("=" * 70)
    
    response = requests.post(f"{BASE_URL}/analyze", json=sample_lab_values)
    
    if response.status_code == 200:
        result = response.json()
        print("✓ Success!")
        print(json.dumps(result, indent=2))
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)

def test_disease_classification():
    """Test disease classification only"""
    
    print("\nTesting DISEASE CLASSIFICATION...")
    print("=" * 70)
    
    response = requests.post(f"{BASE_URL}/disease-classification", json=sample_lab_values)
    
    if response.status_code == 200:
        result = response.json()
        print("✓ Success!")
        print(f"Disease: {result.get('disease_type')}")
        print(f"Confidence: {result.get('confidence')}")
    else:
        print(f"❌ Error: {response.status_code}")

def test_trend_forecasting():
    """Test trend forecasting"""
    
    print("\nTesting TREND FORECASTING...")
    print("=" * 70)
    
    response = requests.post(f"{BASE_URL}/trend-forecast", json=sample_lab_values)
    
    if response.status_code == 200:
        result = response.json()
        print("✓ Success!")
        for param, forecast in result.items():
            if 'forecast_30d' in forecast:
                print(f"{param}: {forecast['current']} → {forecast['forecast_90d']} ({forecast['trend']})")
    else:
        print(f"❌ Error: {response.status_code}")

def test_risk_assessment():
    """Test risk assessment"""
    
    print("\nTesting RISK ASSESSMENT...")
    print("=" * 70)
    
    response = requests.post(f"{BASE_URL}/risk-assessment", json=sample_lab_values)
    
    if response.status_code == 200:
        result = response.json()
        print("✓ Success!")
        print(f"Risk Score: {result.get('risk_score')}/100")
        print(f"Risk Level: {result.get('risk_level')}")
        print("\nRecommendations:")
        for rec in result.get('recommendations', []):
            print(f"  • {rec}")
    else:
        print(f"❌ Error: {response.status_code}")

def test_health_check():
    """Test health check"""
    
    print("\nTesting HEALTH CHECK...")
    print("=" * 70)
    
    response = requests.get(f"{BASE_URL}/health")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Status: {result.get('status')}")
        print(f"Message: {result.get('message')}")
    else:
        print(f"❌ Error: {response.status_code}")

if __name__ == "__main__":
    print("Starting API tests...")
    print("Make sure API is running: python -m uvicorn app.main:app --reload\n")
    
    # Test health first
    test_health_check()
    
    # Test individual endpoints
    test_disease_classification()
    test_trend_forecasting()
    test_risk_assessment()
    
    # Test complete analysis
    test_complete_analysis()
    
    print("\n" + "=" * 70)
    print("✅ All tests completed!")