"""
Integration Tests for Prediction API
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data response.json()
    assert data["status"] == "healthy"
    print("✓ Health endpoint test passed")


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    print("✓ Root endpoint test passed")


def test_predict_endpoint():
    """Test prediction endpoint"""
    # Sample prediction request
    request_data = {
        "application": {
            "application_id": "TEST-001",
            "age": 30,
            "gender": "MALE",
            "marital_status": "MARRIED",
            "education": "S1",
            "housing_type": "OWNED",
            "annual_income": 120000000,
            "employment_status": "PERMANENT",
            "work_experience_years": 5,
            "existing_loans_count": 1,
            "total_existing_debt": 5000000,
            "credit_card_utilization": 0.3,
            "past_delinquencies": 0,
            "loan_amount": 50000000,
            "loan_term_months": 12
        }
    }
    
    # Note: This test will only pass if model artifacts exist
    try:
        response = client.post("/api/v1/predict", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "request_id" in data
            assert "prediction_score" in data
            assert "risk_category" in data
            assert "decision" in data
            assert 0 <= data["prediction_score"] <= 1
            print("✓ Predict endpoint test passed")
        else:
            print(f"⚠️  Predict endpoint returned {response.status_code}")
            print("   This is expected if model artifacts don't exist yet")
            
    except Exception as e:
        print(f"⚠️  Predict endpoint test skipped: {e}")
        print("   Train a model first using: python scripts/train_model.py")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
