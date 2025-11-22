"""
Test Suite for Preprocessing Module
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.features.preprocessing import FeatureEngineer, CreditPreprocessor


@pytest.fixture
def sample_data():
    """Generate sample credit data for testing"""
    return pd.DataFrame({
        'age': [30, 25, 40],
        'gender': ['MALE', 'FEMALE', 'MALE'],
        'marital_status': ['MARRIED', 'SINGLE', 'MARRIED'],
        'education': ['S1', 'SMA', 'S2'],
        'housing_type': ['OWNED', 'RENTED', 'OWNED'],
        'annual_income': [120000000, 60000000, 200000000],
        'employment_status': ['PERMANENT', 'CONTRACT', 'PERMANENT'],
        'work_experience_years': [5, 2, 15],
        'existing_loans_count': [1, 0, 2],
        'total_existing_debt': [5000000, 0, 10000000],
        'credit_card_utilization': [0.3, 0.1, 0.5],
        'past_delinquencies': [0, 0, 1],
        'loan_amount': [50000000, 20000000, 100000000],
        'loan_term_months': [12, 6, 24]
    })


def test_feature_engineer(sample_data):
    """Test feature engineering"""
    fe = FeatureEngineer()
    X_transformed = fe.transform(sample_data)
    
    # Check new features are created
    assert 'debt_service_ratio' in X_transformed.columns
    assert 'income_to_loan_ratio' in X_transformed.columns
    assert 'disposable_income' in X_transformed.columns
    assert 'is_high_risk_dsr' in X_transformed.columns
    
    # Check DSR calculation is reasonable
    assert all(X_transformed['debt_service_ratio'] >= 0)
    
    print("✓ Feature engineering tests passed")


def test_credit_preprocessor(sample_data):
    """Test credit preprocessor"""
    preprocessor = CreditPreprocessor()
    
    # Fit and transform
    preprocessor.fit(sample_data)
    X_transformed = preprocessor.transform(sample_data)
    
    # Check output shape
    assert X_transformed.shape[0] == len(sample_data)
    assert X_transformed.shape[1] > 0
    
    # Check feature names
    feature_names = preprocessor.get_feature_names()
    assert len(feature_names) == X_transformed.shape[1]
    
    print(f"✓ Preprocessor tests passed - Output shape: {X_transformed.shape}")


def test_preprocessor_consistency(sample_data):
    """Test that transform is consistent"""
    preprocessor = CreditPreprocessor()
    preprocessor.fit(sample_data)
    
    # Transform twice
    X1 = preprocessor.transform(sample_data)
    X2 = preprocessor.transform(sample_data)
    
    # Should be identical
    np.testing.assert_array_almost_equal(X1, X2)
    
    print("✓ Preprocessor consistency tests passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
