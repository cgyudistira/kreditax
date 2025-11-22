import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # POJK / Credit Risk Features
        # Debt Service Ratio (DSR)
        # Assuming monthly installment is roughly loan_amount / loan_term_months (simplified)
        # In real scenario, use PMT formula.
        # DSR = (Existing Monthly Debt + New Loan Monthly Installment) / Monthly Income
        
        monthly_income = X['annual_income'] / 12
        # Estimate monthly installment for the new loan (simple division for MVP)
        new_installment = X['loan_amount'] / X['loan_term_months']
        
        # Estimate existing monthly debt payment (approx 3% of total debt as min payment)
        existing_monthly_debt = X['total_existing_debt'] * 0.03
        
        X['debt_service_ratio'] = (existing_monthly_debt + new_installment) / monthly_income
        
        # Income to Loan Ratio
        X['income_to_loan_ratio'] = X['annual_income'] / X['loan_amount']
        
        # Disposable Income
        X['disposable_income'] = monthly_income - existing_monthly_debt - new_installment
        
        # Flag High Risk
        X['is_high_risk_dsr'] = (X['debt_service_ratio'] > 0.4).astype(int)
        
        return X

class CreditPreprocessor:
    def __init__(self):
        self.pipeline = None
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
        # Feature Engineering first to get new columns
        fe = FeatureEngineer()
        X_fe = fe.transform(X)
        
        # Define transformers
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.pipeline = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.num_features),
                ('cat', categorical_transformer, self.cat_features)
            ]
        )
        
        self.pipeline.fit(X_fe, y)
        return self
    
    def transform(self, X):
        fe = FeatureEngineer()
        X_fe = fe.transform(X)
        return self.pipeline.transform(X_fe)
    
    def get_feature_names(self):
        # Helper to get feature names after transformation
        if not self.pipeline:
            return []
            
        num_names = self.num_features
        
        cat_encoder = self.pipeline.named_transformers_['cat']['encoder']
        cat_names = cat_encoder.get_feature_names_out(self.cat_features)
        
        return list(num_names) + list(cat_names)
    
    def save(self, path):
        joblib.dump(self, path)
        
    @staticmethod
    def load(path):
        return joblib.load(path)

if __name__ == "__main__":
    # Test run
    df = pd.read_csv("data/sample_credit_data.csv")
    X = df.drop(columns=['is_default', 'application_id'])
    y = df['is_default']
    
    preprocessor = CreditPreprocessor()
    preprocessor.fit(X)
    X_transformed = preprocessor.transform(X)
    
    print("Preprocessing successful.")
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {X_transformed.shape}")
    print(f"Features: {preprocessor.get_feature_names()}")
    
    # Save preprocessor
    import os
    os.makedirs("models", exist_ok=True)
    preprocessor.save("models/preprocessor.joblib")
