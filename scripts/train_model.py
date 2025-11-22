"""
Complete Model Training Script

End-to-end training pipeline from data generation to model deployment.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.features.preprocessing import CreditPreprocessor
from ml.training.train import CreditScoreTrainer
from ml.evaluation.evaluate import ModelEvaluator


def main(
    data_path: str = "data/demo/sample_credit_data.csv",
    model_type: str = "xgboost",
    test_size: float = 0.2,
    cv_folds: int = 5,
    output_dir: str = "ml/artifacts"
):
    """
    Train credit scoring model.
    
    Args:
        data_path: Path to training data CSV
        model_type: 'xgboost' or 'lightgbm'
        test_size: Proportion of data for testing
        cv_folds: Number of cross-validation folds
        output_dir: Directory to save model artifacts
    """
    print("="*70)
    print("KreditaX Model Training Pipeline")
    print("="*70)
    
    # Step 1: Load data
    print("\n[1/6] Loading data...")
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} samples from {data_path}")
    print(f"  Default rate: {df['is_default'].mean():.2%}")
    
    # Step 2: Split features and target
    print("\n[2/6] Preparing features...")
    X = df.drop(columns=['is_default', 'application_id'])
    y = df['is_default']
    print(f"✓ Features: {X.shape[1]} columns")
    print(f"✓ Target distribution: {y.value_counts().to_dict()}")
    
    # Step 3: Preprocessing
    print("\n[3/6] Preprocessing data...")
    preprocessor = CreditPreprocessor()
    preprocessor.fit(X, y)
    X_processed = preprocessor.transform(X)
    print(f"✓ Processed shape: {X_processed.shape}")
    print(f"✓ Total features after encoding: {X_processed.shape[1]}")
    
    # Save preprocessor
    preprocessor_path = Path(output_dir) / "preprocessor.joblib"
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessor.save(preprocessor_path)
    print(f"✓ Preprocessor saved to {preprocessor_path}")
    
    # Step 4: Train-test split
    print("\n[4/6] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"✓ Train set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    
    # Step 5: Train model
    print(f"\n[5/6] Training {model_type.upper()} model...")
    trainer = CreditScoreTrainer(model_type=model_type)
    
    # Cross-validation
    print(f"\nPerforming {cv_folds}-fold cross-validation...")
    cv_scores = trainer.cross_validate(X_processed, y, n_splits=cv_folds)
    print(f"\n✓ CV AUC: {cv_scores['auc_mean']:.4f} ± {cv_scores['auc_std']:.4f}")
    print(f"✓ CV PR-AUC: {cv_scores['pr_auc_mean']:.4f} ± {cv_scores['pr_auc_std']:.4f}")
    
    # Train final model on full training set
    print("\nTraining final model...")
    trainer.train(X_train, y_train, X_test, y_test)
    
    # Save model
    model_path = trainer.save_model(output_dir)
    
    # Create latest symlink
    latest_path = Path(output_dir) / f"{model_type}_model_latest.joblib"
    import shutil
    shutil.copy(model_path, latest_path)
    print(f"✓ Latest model: {latest_path}")
    
    # Step 6: Evaluate model
    print("\n[6/6] Evaluating model...")
    evaluator = ModelEvaluator(
        trainer.model,
        X_test,
        y_test,
        feature_names=preprocessor.get_feature_names()
    )
    
    metrics = evaluator.generate_report(
        output_dir="ml/experiments",
        threshold=0.5
    )
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Model Type:     {model_type.upper()}")
    print(f"Test AUC:       {metrics['auc']:.4f}")
    print(f"Test PR-AUC:    {metrics['pr_auc']:.4f}")
    print(f"Model Path:     {latest_path}")
    print(f"Preprocessor:   {preprocessor_path}")
    print("="*70)
    
    # Check if meets MVP criteria
    if metrics['auc'] >= 0.78:
        print("✅ Model meets MVP criteria (AUC >= 0.78)")
    else:
        print(f"⚠️  Model AUC ({metrics['auc']:.4f}) below MVP target (0.78)")
        print("   Consider: more data, feature engineering, or hyperparameter tuning")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train KreditaX credit scoring model")
    parser.add_argument("--data", default="data/demo/sample_credit_data.csv", help="Path to training data")
    parser.add_argument("--model", default="xgboost", choices=["xgboost", "lightgbm"], help="Model type")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set proportion")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--output", default="ml/artifacts", help="Output directory")
    
    args = parser.parse_args()
    
    main(
        data_path=args.data,
        model_type=args.model,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        output_dir=args.output
    )
