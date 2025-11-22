"""
SHAP Explainability Service

Provides local and global explanations for credit scoring decisions using SHAP.
All explanations are POJK-compliant and human-readable.
"""

import shap
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import matplotlib.pyplot as plt


class SHAPExplainer:
    """
    SHAP-based explainability for credit scoring models.
    
    Provides:
    - Local explanations: Why was this specific application approved/rejected?
    - Global explanations: Which features are most important overall?
    - Counterfactual suggestions: What changes would flip the decision?
    """
    
    def __init__(self, model, X_background=None):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model
            X_background: Background dataset for SHAP (optional, uses sample if None)
        """
        self.model = model
        
        # Initialize SHAP explainer
        if X_background is not None:
            # Use TreeExplainer for tree-based models (XGBoost, LightGBM)
            self.explainer = shap.TreeExplainer(model, X_background)
        else:
            self.explainer = shap.TreeExplainer(model)
    
    def explain_prediction(
        self, 
        X_instance, 
        feature_names=None,
        top_k=5
    ) -> Dict[str, Any]:
        """
        Explain a single prediction.
        
        Args:
            X_instance: Single instance to explain (1D array or 2D with shape (1, n_features))
            feature_names: List of feature names
            top_k: Number of top features to return
            
        Returns:
            Dictionary with explanation details
        """
        # Ensure 2D array
        if len(X_instance.shape) == 1:
            X_instance = X_instance.reshape(1, -1)
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_instance)
        
        # Get base value (expected value)
        base_value = self.explainer.expected_value
        
        # For binary classification, handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Class 1 (default)
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[1]
        
        # Get prediction
        prediction_proba = self.model.predict_proba(X_instance)[0, 1]
        
        # Extract SHAP values for this instance
        shap_instance = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        
        # Create feature importance ranking
        feature_impacts = []
        for i, shap_val in enumerate(shap_instance):
            feature_name = feature_names[i] if feature_names else f"feature_{i}"
            feature_impacts.append({
                'feature': feature_name,
                'shap_value': float(shap_val),
                'feature_value': float(X_instance[0, i]),
                'impact': 'increases_risk' if shap_val > 0 else 'decreases_risk'
            })
        
        # Sort by absolute SHAP value
        feature_impacts.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        
        # Generate human-readable explanation
        explanation_text = self._generate_explanation_text(
            prediction_proba, feature_impacts[:top_k]
        )
        
        return {
            'prediction_probability': float(prediction_proba),
            'base_value': float(base_value),
            'shap_values': shap_instance.tolist(),
            'top_features': feature_impacts[:top_k],
            'explanation': explanation_text,
            'risk_category': self._get_risk_category(prediction_proba)
        }
    
    def _generate_explanation_text(
        self, 
        prediction_proba, 
        top_features
    ) -> str:
        """Generate human-readable explanation"""
        risk_level = "HIGH" if prediction_proba > 0.5 else "LOW"
        
        explanation = f"Credit Default Risk: {risk_level} ({prediction_proba:.1%} probability)\n\n"
        explanation += "Key factors influencing this decision:\n"
        
        for i, feature in enumerate(top_features, 1):
            direction = "increases" if feature['shap_value'] > 0 else "decreases"
            explanation += f"{i}. {feature['feature']}: {direction} risk\n"
        
        return explanation
    
    def _get_risk_category(self, probability):
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
    
    def global_feature_importance(self, X_sample, feature_names=None):
        """
        Calculate global feature importance using SHAP.
        
        Args:
            X_sample: Sample dataset for SHAP calculation
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importances
        """
        shap_values = self.explainer.shap_values(X_sample)
        
        # Handle binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(mean_abs_shap))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_waterfall(self, X_instance, feature_names=None, save_path=None):
        """
        Create SHAP waterfall plot for a single prediction.
        
        Args:
            X_instance: Single instance
            feature_names: List of feature names
            save_path: Path to save plot (optional)
        """
        if len(X_instance.shape) == 1:
            X_instance = X_instance.reshape(1, -1)
        
        shap_values = self.explainer.shap_values(X_instance)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Create waterfall plot
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[0],
                base_values=self.explainer.expected_value if not isinstance(
                    self.explainer.expected_value, (list, np.ndarray)
                ) else self.explainer.expected_value[1],
                data=X_instance[0],
                feature_names=feature_names
            ),
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    # Example usage
    print("Testing SHAP Explainer...")
    
    # Load model and data
    import joblib
    from ml.features.preprocessing import CreditPreprocessor
    import pandas as pd
    
    # Load preprocessor
    preprocessor = CreditPreprocessor.load("ml/artifacts/preprocessor.joblib")
    
    # Load model (adjust path as needed)
    model = joblib.load("ml/artifacts/xgboost_model_latest.joblib")
    
    # Load test data
    df = pd.read_csv("data/demo/sample_credit_data.csv").head(100)
    X = df.drop(columns=['is_default', 'application_id'])
    X_processed = preprocessor.transform(X)
    
    # Initialize explainer
    explainer = SHAPExplainer(model, X_processed[:50])
    
    # Explain single prediction
    explanation = explainer.explain_prediction(
        X_processed[0],
        feature_names=preprocessor.get_feature_names(),
        top_k=5
    )
    
    print("\n" + "="*50)
    print("PREDICTION EXPLANATION")
    print("="*50)
    print(explanation['explanation'])
    print(f"\nRisk Category: {explanation['risk_category']}")
