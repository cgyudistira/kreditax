"""
Model Training Module

Trains XGBoost and LightGBM models for credit scoring with:
- Cross-validation
- Hyperparameter tuning
- Model versioning
- Experiment tracking
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import json
from datetime import datetime
from pathlib import Path


class CreditScoreTrainer:
    """
    Credit scoring model trainer with support for XGBoost and LightGBM.
    
    Features:
    - Handles class imbalance with scale_pos_weight
    - Cross-validation for robust evaluation
    - Model versioning and artifact saving
    """
    
    def __init__(self, model_type="xgboost", random_state=42):
        """
        Initialize trainer.
        
        Args:
            model_type: 'xgboost' or 'lightgbm'
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.cv_scores = None
        
    def get_default_params(self, scale_pos_weight=1.0):
        """
        Get default hyperparameters for the model.
        
        Args:
            scale_pos_weight: Weight for positive class (for imbalanced data)
            
        Returns:
            Dictionary of hyperparameters
        """
        if self.model_type == "xgboost":
            return {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'scale_pos_weight': scale_pos_weight,
                'random_state': self.random_state,
                'tree_method': 'hist',
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        else:  # lightgbm
            return {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'objective': 'binary',
                'metric': 'auc',
                'scale_pos_weight': scale_pos_weight,
                'random_state': self.random_state,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'verbose': -1
            }
    
    def train(self, X_train, y_train, X_val=None, y_val=None, params=None):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            params: Hyperparameters (optional, uses defaults if None)
            
        Returns:
            Trained model
        """
        # Calculate class imbalance ratio
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        # Use provided params or defaults
        if params is None:
            params = self.get_default_params(scale_pos_weight)
        
        self.best_params = params
        
        # Train model
        if self.model_type == "xgboost":
            self.model = xgb.XGBClassifier(**params)
            
            if X_val is not None and y_val is not None:
                eval_set = [(X_train, y_train), (X_val, y_val)]
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    verbose=False
                )
            else:
                self.model.fit(X_train, y_train)
                
        else:  # lightgbm
            self.model = lgb.LGBMClassifier(**params)
            
            if X_val is not None and y_val is not None:
                eval_set = [(X_train, y_train), (X_val, y_val)]
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    callbacks=[lgb.log_evaluation(0)]
                )
            else:
                self.model.fit(X_train, y_train)
        
        return self.model
    
    def cross_validate(self, X, y, n_splits=5, params=None):
        """
        Perform stratified k-fold cross-validation.
        
        Args:
            X: Features
            y: Labels
            n_splits: Number of CV folds
            params: Hyperparameters (optional)
            
        Returns:
            Dictionary with CV metrics
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        auc_scores = []
        pr_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train_fold = X[train_idx]
            y_train_fold = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]
            X_val_fold = X[val_idx]
            y_val_fold = y.iloc[val_idx] if isinstance(y, pd.Series) else y[val_idx]
            
            # Train on fold
            self.train(X_train_fold, y_train_fold, params=params)
            
            # Predict on validation fold
            y_pred_proba = self.model.predict_proba(X_val_fold)[:, 1]
            
            # Calculate metrics
            auc = roc_auc_score(y_val_fold, y_pred_proba)
            pr = average_precision_score(y_val_fold, y_pred_proba)
            
            auc_scores.append(auc)
            pr_scores.append(pr)
            
            print(f"Fold {fold + 1}/{n_splits} - AUC: {auc:.4f}, PR-AUC: {pr:.4f}")
        
        self.cv_scores = {
            'auc_mean': np.mean(auc_scores),
            'auc_std': np.std(auc_scores),
            'pr_auc_mean': np.mean(pr_scores),
            'pr_auc_std': np.std(pr_scores),
            'auc_scores': auc_scores,
            'pr_scores': pr_scores
        }
        
        return self.cv_scores
    
    def save_model(self, output_dir="ml/artifacts", version=None):
        """
        Save model and metadata.
        
        Args:
            output_dir: Directory to save artifacts
            version: Model version (auto-generated if None)
            
        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate version string
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_filename = f"{self.model_type}_model_{version}.joblib"
        model_path = output_path / model_filename
        joblib.dump(self.model, model_path)
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'params': self.best_params,
            'cv_scores': self.cv_scores
        }
        
        metadata_filename = f"{self.model_type}_metadata_{version}.json"
        metadata_path = output_path / metadata_filename
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Model saved to {model_path}")
        print(f"✓ Metadata saved to {metadata_path}")
        
        return str(model_path)
    
    @staticmethod
    def load_model(model_path):
        """Load a saved model"""
        return joblib.load(model_path)


if __name__ == "__main__":
    # Example training workflow
    print("Loading preprocessed data...")
    
    # Load data
    from ml.features.preprocessing import CreditPreprocessor
    
    df = pd.read_csv("data/demo/sample_credit_data.csv")
    X = df.drop(columns=['is_default', 'application_id'])
    y = df['is_default']
    
    # Preprocess
    preprocessor = CreditPreprocessor()
    preprocessor.fit(X, y)
    X_processed = preprocessor.transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train XGBoost
    print("\n" + "="*50)
    print("Training XGBoost Model")
    print("="*50)
    
    trainer_xgb = CreditScoreTrainer(model_type="xgboost")
    trainer_xgb.train(X_train, y_train)
    
    # Evaluate
    y_pred_proba = trainer_xgb.model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n✓ Test AUC: {test_auc:.4f}")
    
    # Save model
    trainer_xgb.save_model()
