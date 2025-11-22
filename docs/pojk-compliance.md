# POJK Compliance Documentation

## Overview
KreditaX is designed to comply with Indonesian banking regulations (Peraturan Otoritas Jasa Keuangan - POJK) for AI-based credit scoring systems.

## Key Compliance Features

### 1. Auditability
**Requirement**: All credit decisions must be auditable with complete audit trails.

**Implementation**:
- Every prediction is logged with a unique `request_id`
- Audit logs include:
  - Timestamp
  - Model version used
  - Masked input features (PII protected)
  - Prediction score
  - Risk category
  - Decision outcome (APPROVE/REJECT)
  - Explanation summary

**Location**: `app/services/audit_service.py`

**Access**: Via `/api/v1/audit-log` endpoint

###  2. Explainability
**Requirement**: Credit decisions must be explainable to customers and regulators.

**Implementation**:
- SHAP (SHapley Additive exPlanations) integration
- Local explanations for each prediction
- Top contributing factors identified
- Human-readable explanation text
- Global feature importance available

**Location**: `app/services/explain_service.py`

**Access**: Explanations included in `/api/v1/predict` response

### 3. Feature Logging
**Requirement**: All features used in decision-making must be logged.

**Implementation**:
- Complete feature list stored with preprocessor
- Feature engineering steps documented
- POJK-specific features:
  - Debt Service Ratio (DSR)
  - Income-to-Loan Ratio
  - Disposable Income
  - Risk flags based on regulatory thresholds

**Location**: `ml/features/preprocessing.py`

### 4. Data Privacy
**Requirement**: PII must be protected in logs and storage.

**Implementation**:
- PII fields masked before logging:
  - `application_id`: Hashed with SHA-256
  - Financial amounts: Rounded to nearest 10M IDR
- Only statistical properties retained
- Original PII never stored in audit logs

**Location**: `app/services/audit_service.py` (`_mask_pii` method)

### 5. Model Versioning
**Requirements**: Model version must be tracked for each decision.

**Implementation**:
- Model artifacts include version metadata
- Version logged with each prediction
- Timestamp-based versioning
- Ability to rollback to previous versions

**Location**: `ml/training/train.py` (`save_model` method)

### 6. Decision Thresholds
**Requirement**: Clear, documented decision thresholds.

**Implementation**:
- Configurable risk threshold (default: 0.5)
- Risk categories: VERY_LOW, LOW, MEDIUM, HIGH, VERY_HIGH
- Threshold can be adjusted per business rules
- Documented in configuration

**Location**: `app/core/config.py`

### 7. Fairness and Bias Monitoring
**Recommendation**: Monitor for demographic bias.

**Implementation**:
- Demographic features (age, gender, education) tracked separately
- Audit logs enable post-hoc bias analysis
- Feature importance helps identify discriminatory patterns

**Future Enhancement**: Add automated bias detection in model evaluation

## Regulatory Checklist

- [x] Audit trail for all decisions
- [x] Explainable predictions (SHAP)
- [x] Feature logging
- [x] PII masking
- [x] Model versioning
- [x] Configurable decision thresholds
- [x] Export capability for audit logs
- [ ] Automated bias detection (future)
- [ ] Periodic model retraining alerts (future)

## Audit Log Retention

**Current**: CSV-based storage in `data/audit_logs.csv`

**Recommendation for Production**:
- Database storage (PostgreSQL)
- Retention policy: 7 years (per banking regulations)
- Encrypted storage
- Regular backups
- Immutable logs (append-only)

## Compliance Reports

To generate compliance reports:

```bash
# Export audit logs
curl http://localhost:8000/api/v1/audit-log/export -o audit_report.csv

# View recent decisions
curl http://localhost:8000/api/v1/audit-log?limit=100
```

## Contact

For compliance questions or audit requests:
- Technical: KreditaX Development Team
- Regulatory: Compliance Officer
