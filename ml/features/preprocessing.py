"""
Credit Data Preprocessing and Feature Engineering

POJK-compliant feature engineering including:
- Debt Service Ratio (DSR) calculation
- Income-to-loan ratios
- Risk flags

All transformations are designed to be auditable and explainable.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    POJK-Compliant Feature Engineering
    
    Creates derived features for credit risk assessment:
    - Debt Service Ratio (DSR): Monthly debt obligations / Monthly income
    - Income to Loan Ratio: How many times income covers the loan
    - Disposable Income: Income after all debt obligations
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        """Fit method (stateless transformer)"""
        return self
    
    def transform(self, X):
        """
        Transform input data by adding engineered features.
        
        Args:
            X: Input dataframe with raw features
            
        Returns:
            Transformed dataframe with engineered features
        """
        X = X.copy()
        
        # Calculate monthly income
        monthly_income = X['annual_income'] / 12
        
        # Estimate monthly installment for the new loan (simplified)
        # In production, use PMT formula: PMT = P * (r * (1 + r)^n) / ((1 + r)^n - 1)
        new_installment = X['loan_amount'] / X['loan_term_months']
        
        # Estimate existing monthly debt payment (approx 3% of total debt as minimum payment)
        existing_monthly_debt = X['total_existing_debt'] * 0.03
        
        # POJK Feature 1: Debt Service Ratio
        # DSR = (Existing Monthly Debt + New Loan Monthly Installment) / Monthly Income
        X['debt_service_ratio'] = (existing_monthly_debt + new_installment) / monthly_income
        
        # POJK Feature 2: Income to Loan Ratio
        X['income_to_loan_ratio'] = X['annual_income'] / X['loan_amount']
        
        # POJK Feature 3: Disposable Income
        X['disposable_income'] = monthly_income - existing_monthly_debt - new_installment
        
        # POJK Feature 4: High Risk Flag (DSR > 40% is typically considered risky)
        X['is_high_risk_dsr'] = (X['debt_service_ratio'] > 0.4).astype(int)
        
        return X


class CreditPreprocessor:
    """
    Complete preprocessing pipeline for credit data.
    
    Handles:
    - Feature engineering (DSR, ratios)
    - Missing value imputation
    - Categorical encoding (one-hot)
    - Numerical scaling (standardization)
    """
    
    def __init__(self):
        self.pipeline = None
        self.feature_engineer = FeatureEngineer()
        
        # Define feature groups
        self.cat_features = [
            'gender', 'marital_status', 'education', 
            'housing_type', 'employment_status'
        ]
        
        self.num_features = [
            'age', 'annual_income', 'work_experience_years',
            'existing_loans_count', 'total_existing_debt',
            'credit_card_utilization', 'past_delinquencies',
            'loan_amount', 'loan_term_months',
            'debt_service_ratio', 'income_to_loan_ratio', 'disposable_income'
        ]
        
    def fit(self, X, y=None):
        """
        Fit the preprocessing pipeline.
        
        Args:
            X: Training data
            y: Target variable (optional)
            
        Returns:
            self
        """
        # Feature Engineering first to get new columns
        X_fe = self.feature_engineer.transform(X)
        
        # Define transformers
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine transformers
        self.pipeline = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.num_features),
                ('cat', categorical_transformer, self.cat_features)
            ]
        )
        
        self.pipeline.fit(X_fe, y)
        return self
    
    def transform(self, X):
        """
        Transform input data using fitted pipeline.
        
        Args:
            X: Input data
            
        Returns:
            Transformed numpy array
        """
        X_fe = self.feature_engineer.transform(X)
        return self.pipeline.transform(X_fe)
    
    def get_feature_names(self):
        """
        Get feature names after transformation.
        
        Returns:
            List of feature names
        """
        if not self.pipeline:
            return []
            
        num_names = self.num_features
        
        cat_encoder = self.pipeline.named_transformers_['cat']['encoder']
        cat_names = cat_encoder.get_feature_names_out(self.cat_features)
        
        return list(num_names) + list(cat_names)
    
    def save(self, path):
        """Save preprocessor to disk"""
        joblib.dump(self, path)
        
    @staticmethod
    def load(path):
        """Load preprocessor from disk"""
        return joblib.load(path)


if __name__ == "__main__":
    # Test preprocessing pipeline
    print("Testing preprocessing pipeline...")
    
    df = pd.read_csv("data/demo/sample_credit_data.csv")
    X = df.drop(columns=['is_default', 'application_id'])
    y = df['is_default']
    
    preprocessor = CreditPreprocessor()
    preprocessor.fit(X)
    X_transformed = preprocessor.transform(X)
    
    print("✓ Preprocessing successful")
    print(f"✓ Input shape: {X.shape}")
    print(f"✓ Output shape: {X_transformed.shape}")
    print(f"✓ Number of features: {len(preprocessor.get_feature_names())}")
    
    # Save preprocessor
    import os
    os.makedirs("ml/artifacts", exist_ok=True)
    preprocessor.save("ml/artifacts/preprocessor.joblib")
    print("✓ Preprocessor saved to ml/artifacts/preprocessor.joblib")
