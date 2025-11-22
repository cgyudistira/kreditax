"""
Audit Logging Service

POJK-compliant audit logging for all credit scoring decisions.
Logs are immutable and include masked PII for privacy.
"""

import pandas as pd
import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import json


class AuditLogger:
    """
    Audit logger for credit scoring decisions.
    
    Ensures POJK compliance by logging:
    - Request ID and timestamp
    - Masked input features (PII protection)
    - Model version
    - Prediction score
    - Explanation summary
    - Decision outcome
    """
    
    def __init__(self, log_path: str = "data/audit_logs.csv"):
        """
        Initialize audit logger.
        
        Args:
            log_path: Path to audit log CSV file
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create log file with headers if it doesn't exist
        if not self.log_path.exists():
            self._initialize_log_file()
    
    def _initialize_log_file(self):
        """Create log file with headers"""
        headers = [
            'request_id', 'timestamp', 'model_version', 
            'prediction_score', 'risk_category', 
            'decision', 'explanation_summary',
            'masked_features_hash', 'user_id'
        ]
        df = pd.DataFrame(columns=headers)
        df.to_csv(self.log_path, index=False)
    
    def _mask_pii(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mask PII fields in features.
        
        Args:
            features: Raw feature dictionary
            
        Returns:
            Masked feature dictionary
        """
        masked = features.copy()
        
        # Fields to mask completely
        pii_fields = ['application_id', 'name', 'email', 'phone', 'address']
        
        for field in pii_fields:
            if field in masked:
                masked[field] = self._hash_value(str(masked[field]))
        
        # Mask partially (keep statistical properties)
        partial_mask_fields = ['annual_income', 'loan_amount', 'total_existing_debt']
        for field in partial_mask_fields:
            if field in masked:
                # Round to nearest 10M for privacy
                value = masked[field]
                masked[field] = round(value / 10000000) * 10000000
        
        return masked
    
    def _hash_value(self, value: str) -> str:
        """Hash a value using SHA256"""
        return hashlib.sha256(value.encode()).hexdigest()[:16]
    
    def log_prediction(
        self,
        request_id: Optional[str],
        features: Dict[str, Any],
        prediction_score: float,
        risk_category: str,
        explanation: Dict[str, Any],
        model_version: str = "1.0.0",
        user_id: Optional[str] = None
    ) -> str:
        """
        Log a credit scoring prediction.
        
        Args:
            request_id: Unique request ID (generated if None)
            features: Input features
            prediction_score: Model prediction probability
            risk_category: Risk category (LOW, MEDIUM, HIGH, etc.)
            explanation: SHAP explanation dictionary
            model_version: Model version used
            user_id: User who made the request (optional)
            
        Returns:
            Request ID
        """
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        # Mask PII
        masked_features = self._mask_pii(features)
        
        # Hash features for audit trail
        features_hash = self._hash_value(json.dumps(masked_features, sort_keys=True))
        
        # Determine decision
        decision = "REJECT" if prediction_score > 0.5 else "APPROVE"
        
        # Extract explanation summary
        explanation_summary = explanation.get('explanation', 'No explanation available')
        if len(explanation_summary) > 200:
            explanation_summary = explanation_summary[:200] + "..."
        
        # Create log entry
        log_entry = {
            'request_id': request_id,
            'timestamp': datetime.now().isoformat(),
            'model_version': model_version,
            'prediction_score': round(prediction_score, 4),
            'risk_category': risk_category,
            'decision': decision,
            'explanation_summary': explanation_summary,
            'masked_features_hash': features_hash,
            'user_id': user_id or 'anonymous'
        }
        
        # Append to log file
        df = pd.DataFrame([log_entry])
        df.to_csv(self.log_path, mode='a', header=False, index=False)
        
        return request_id
    
    def get_logs(
        self, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Retrieve audit logs.
        
        Args:
            start_date: Start date filter (ISO format)
            end_date: End date filter (ISO format)
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with audit logs
        """
        if not self.log_path.exists():
            return pd.DataFrame()
        
        df = pd.read_csv(self.log_path)
        
        # Filter by date range
        if start_date:
            df = df[df['timestamp'] >= start_date]
        if end_date:
            df = df[df['timestamp'] <= end_date]
        
        # Limit results
        return df.tail(limit)
    
    def export_logs(self, output_path: str):
        """
        Export audit logs to CSV.
        
        Args:
            output_path: Path to export file
        """
        if not self.log_path.exists():
            raise FileNotFoundError("No audit logs found")
        
        df = pd.read_csv(self.log_path)
        df.to_csv(output_path, index=False)
        print(f"✓ Audit logs exported to {output_path}")


if __name__ == "__main__":
    # Test audit logger
    logger = AuditLogger()
    
    # Log a sample prediction
    request_id = logger.log_prediction(
        request_id=None,
        features={
            'application_id': 'APP-12345',
            'age': 30,
            'annual_income': 120000000,
            'loan_amount': 50000000
        },
        prediction_score=0.25,
        risk_category='LOW',
        explanation={'explanation': 'Low risk due to stable income and good credit history'},
        model_version='1.0.0',
        user_id='test_user'
    )
    
    print(f"✓ Logged prediction with request ID: {request_id}")
    
    # Retrieve logs
    logs = logger.get_logs(limit=5)
    print(f"✓ Retrieved {len(logs)} audit log entries")
