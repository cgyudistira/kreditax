"""
Model Inference Service

Handles model loading and prediction with preprocessing.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
from app.core.config import get_settings


class ModelService:
    """
    Model inference service.
    
    Handles:
    - Model and preprocessor loading
    - Prediction with preprocessing
    - Caching for performance
    """
    
    def __init__(self):
        """Initialize model service"""
        self.settings = get_settings()
        self.model = None
        self.preprocessor = None
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load model and preprocessor"""
        try:
            model_path = Path(self.settings.model_path)
            preprocessor_path = Path(self.settings.preprocessor_path)
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")
            if not preprocessor_path.exists():
                raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
            
            self.model = joblib.load(model_path)
            self.preprocessor = joblib.load(preprocessor_path)
            
            print(f"✓ Model loaded from {model_path}")
            print(f"✓ Preprocessor loaded from {preprocessor_path}")
            
        except Exception as e:
            print(f"✗ Error loading model artifacts: {e}")
            raise
    
    def preprocess(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess input features.
        
        Args:
            features: Dictionary of input features
            
        Returns:
            Preprocessed feature array
        """
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Remove application_id if present
        if 'application_id' in df.columns:
            df = df.drop(columns=['application_id'])
        
        # Remove is_default if present
        if 'is_default' in df.columns:
            df = df.drop(columns=['is_default'])
        
        # Transform using preprocessor
        X_processed = self.preprocessor.transform(df)
        
        return X_processed
    
    def predict(self, features: Dict[str, Any]) -> Tuple[float, str]:
        """
        Make prediction.
        
        Args:
            features: Dictionary of input features
            
        Returns:
            Tuple of (probability, risk_category)
        """
        # Preprocess
        X_processed = self.preprocess(features)
        
        # Predict
        prediction_proba = self.model.predict_proba(X_processed)[0, 1]
        
        # Categorize risk
        risk_category = self._categorize_risk(prediction_proba)
        
        return float(prediction_proba), risk_category
    
    def _categorize_risk(self, probability: float) -> str:
        """Categorize risk level"""
        if probability < 0.2:
            return "VERY_LOW"
        elif probability < 0.4:
            return "LOW"
        elif probability < 0.6:
            return "MEDIUM"
        elif probability < 0.8:
            return "HIGH"
        else:
            return "VERY_HIGH"
    
    def get_feature_names(self):
        """Get feature names from preprocessor"""
        return self.preprocessor.get_feature_names()


# Global model service instance (singleton)
_model_service = None


def get_model_service() -> ModelService:
    """Get or create model service instance"""
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service
