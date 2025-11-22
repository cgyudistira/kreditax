"""
Model Evaluation Module

Comprehensive evaluation metrics for credit scoring models:
- ROC-AUC, PR-AUC
- Calibration plots
- Confusion matrix
- Feature importance
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ModelEvaluator:
    """
    Comprehensive model evaluation for credit scoring.
    
    Provides metrics and visualizations for POJK compliance:
    - Performance metrics (AUC, Precision, Recall)
    - Calibration assessment
    - Feature importance
    """
    
    def __init__(self, model, X_test, y_test, feature_names=None):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            feature_names: List of feature names (optional)
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.y_pred_proba = model.predict_proba(X_test)[:, 1]
        self.y_pred = model.predict(X_test)
        
    def calculate_metrics(self, threshold=0.5):
        """
        Calculate all evaluation metrics.
        
        Args:
            threshold: Classification threshold
            
        Returns:
            Dictionary of metrics
        """
        # Probability-based metrics
        auc = roc_auc_score(self.y_test, self.y_pred_proba)
        pr_auc = average_precision_score(self.y_test, self.y_pred_proba)
        brier = brier_score_loss(self.y_test, self.y_pred_proba)
        
        # Threshold-based metrics
        y_pred_thresh = (self.y_pred_proba >= threshold).astype(int)
        cm = confusion_matrix(self.y_test, y_pred_thresh)
        
        tn, fp, fn, tp = cm.ravel()
        
        metrics = {
            'auc': auc,
            'pr_auc': pr_auc,
            'brier_score': brier,
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'confusion_matrix': cm.tolist(),
            'threshold': threshold
        }
        
        return metrics
    
    def plot_roc_curve(self, save_path=None):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        auc = roc_auc_score(self.y_test, self.y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Credit Scoring Model')
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ ROC curve saved to {save_path}")
        
        plt.close()
    
    def plot_precision_recall_curve(self, save_path=None):
        """Plot Precision-Recall curve"""
        precision, recall, thresholds = precision_recall_curve(self.y_test, self.y_pred_proba)
        pr_auc = average_precision_score(self.y_test, self.y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - Credit Scoring Model')
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ PR curve saved to {save_path}")
        
        plt.close()
    
    def plot_calibration_curve(self, n_bins=10, save_path=None):
        """Plot calibration curve"""
        fraction_of_positives, mean_predicted_value = calibration_curve(
            self.y_test, self.y_pred_proba, n_bins=n_bins
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, marker='o', linewidth=2, label='Model')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Plot - Credit Scoring Model')
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Calibration curve saved to {save_path}")
        
        plt.close()
    
    def plot_confusion_matrix(self, threshold=0.5, save_path=None):
        """Plot confusion matrix"""
        y_pred_thresh = (self.y_pred_proba >= threshold).astype(int)
        cm = confusion_matrix(self.y_test, y_pred_thresh)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Good', 'Default'],
                    yticklabels=['Good', 'Default'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix (Threshold = {threshold:.2f})')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix saved to {save_path}")
        
        plt.close()
    
    def plot_feature_importance(self, top_n=20, save_path=None):
        """Plot feature importance"""
        if not hasattr(self.model, 'feature_importances_'):
            print("Model does not support feature importance")
            return
        
        importance = self.model.feature_importances_
        
        if self.feature_names is not None:
            feature_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            })
        else:
            feature_df = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(importance))],
                'importance': importance
            })
        
        feature_df = feature_df.sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(feature_df)), feature_df['importance'])
        plt.yticks(range(len(feature_df)), feature_df['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Feature importance plot saved to {save_path}")
        
        plt.close()
        
        return feature_df
    
    def generate_report(self, output_dir="ml/experiments", threshold=0.5):
        """
        Generate comprehensive evaluation report.
        
        Args:
            output_dir: Directory to save plots and metrics
            threshold: Classification threshold
            
        Returns:
            Dictionary of metrics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Calculate metrics
        metrics = self.calculate_metrics(threshold)
        
        # Generate plots
        self.plot_roc_curve(output_path / "roc_curve.png")
        self.plot_precision_recall_curve(output_path / "pr_curve.png")
        self.plot_calibration_curve(save_path=output_path / "calibration_curve.png")
        self.plot_confusion_matrix(threshold, save_path=output_path / "confusion_matrix.png")
        feature_importance = self.plot_feature_importance(save_path=output_path / "feature_importance.png")
        
        # Save metrics
        import json
        metrics_path = output_path / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ Metrics saved to {metrics_path}")
        
        # Print summary
        print("\n" + "="*50)
        print("MODEL EVALUATION SUMMARY")
        print("="*50)
        print(f"AUC-ROC:        {metrics['auc']:.4f}")
        print(f"PR-AUC:         {metrics['pr_auc']:.4f}")
        print(f"Brier Score:    {metrics['brier_score']:.4f}")
        print(f"Accuracy:       {metrics['accuracy']:.4f}")
        print(f"Precision:      {metrics['precision']:.4f}")
        print(f"Recall:         {metrics['recall']:.4f}")
        print(f"F1-Score:       {metrics['f1_score']:.4f}")
        print("="*50)
        
        return metrics
