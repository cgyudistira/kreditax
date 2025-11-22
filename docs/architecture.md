# System Architecture

## Overview

KreditaX is a microservices-based credit scoring system with three main layers:

1. **Data Layer**: Data storage, preprocessing, and feature engineering
2. **ML Layer**: Model training, evaluation, and inference
3. **API Layer**: RESTful API for predictions and audit logs

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                       Client Applications                    │
│         (Web Dashboard, Mobile App, Banking System)         │
└────────────────┬────────────────────────────────────────────┘
                 │ HTTP/HTTPS
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐   │
│  │  /predict  │  │  /explain  │  │    /audit-log      │   │
│  └──────┬─────┘  └──────┬─────┘  └─────────┬──────────┘   │
│         │               │                    │               │
│         └───────┬───────┘                    │               │
│                 ▼                            ▼               │
│    ┌─────────────────────┐       ┌──────────────────┐      │
│    │  Model Service      │       │  Audit Service   │      │
│    │  - Preprocessing    │       │  - PII Masking   │      │
│    │  - Inference        │       │  - Log Storage   │      │
│    └─────────┬───────────┘       └──────────────────┘      │
└──────────────┼─────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│                      ML Components                           │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │Preprocessor │  │ XGBoost/LGBM │  │ SHAP Explainer  │   │
│  │  (Joblib)   │  │   (Joblib)   │  │                 │   │
│  └─────────────┘  └──────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│                        Data Layer                            │
│  ┌──────────┐  ┌───────────┐  ┌────────────────────────┐  │
│  │   Raw    │  │ Processed │  │    Audit Logs (CSV)    │  │
│  │   Data   │  │   Data    │  │  (Future: PostgreSQL)  │  │
│  └──────────┘  └───────────┘  └────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. API Layer (`app/`)

**Purpose**: Expose credit scoring functionality via REST API

**Components**:
- `main.py`: FastAPI application entry point
- `api/v1/`: API endpoint handlers
  - `predict.py`: Credit score prediction
  - `health.py`: Health checks
  - `audit.py`: Audit log retrieval
- `core/`: Configuration and utilities
  - `config.py`: Settings management
- `schemas/`: Pydantic models for request/response validation
- `services/`: Business logic
  - `model_service.py`: Model loading and inference
  - `explain_service.py`: SHAP explanations
  - `audit_service.py`: Audit logging

**Technology Stack**:
- FastAPI: Web framework
- Pydantic: Data validation
- Uvicorn: ASGI server

### 2. ML Layer (`ml/`)

**Purpose**: Machine learning pipeline for credit scoring

**Components**:
- `features/preprocessing.py`: Feature engineering and preprocessing
- `training/train.py`: Model training with XGBoost/LightGBM
- `evaluation/evaluate.py`: Model evaluation metrics
- `artifacts/`: Saved models and preprocessors (gitignored)

**ML Pipeline**:
1. **Data Ingestion**: Load credit application data
2. **Feature Engineering**: Calculate DSR, income ratios, risk flags
3. **Preprocessing**: Encode categoricals, scale numerics
4. **Training**: XGBoost/LightGBM with cross-validation
5. **Evaluation**: AUC-ROC, calibration, feature importance
6. **Serialization**: Save model + preprocessor as joblib files

**Technology Stack**:
- XGBoost / LightGBM: Gradient boosting models
- Scikit-learn: Preprocessing and metrics
- SHAP: Model explainability
- Joblib: Model serialization

### 3. Data Layer (`data/`)

**Purpose**: Data storage and management

**Structure**:
- `raw/`: Original uploaded data
- `processed/`: Cleaned and preprocessed data
- `demo/`: Sample datasets for testing
- `audit_logs.csv`: Decision audit trail

**Future Enhancements**:
- PostgreSQL for audit logs
- S3/MinIO for model artifacts
- Data versioning with DVC

### 4. Scripts Layer (`scripts/`)

**Purpose**: Utility scripts for data generation and training

**Scripts**:
- `generate_data.py`: Generate synthetic credit data
- `train_model.py`: Complete training pipeline

## Data Flow

### Prediction Request Flow

```
1. Client sends POST /api/v1/predict
   ↓
2. Pydantic validates request schema
   ↓
3. ModelService preprocesses features
   ↓
4. Model predicts default probability
   ↓
5. SHAPExplainer generates explanation
   ↓
6. AuditLogger logs decision (PII masked)
   ↓
7. Response returned to client
```

### Training Flow

```
1. Generate/load training data
   ↓
2. FeatureEngineer adds POJK features
   ↓
3. CreditPreprocessor fits on training data
   ↓
4. Train XGBoost/LightGBM model
   ↓
5. Cross-validation for robust metrics
   ↓
6. Evaluate on test set
   ↓
7. Save model + preprocessor artifacts
   ↓
8. Generate evaluation report
```

## Security

### API Security
- **Authentication**: JWT tokens (to be implemented)
- **Authorization**: Role-based access control (to be implemented)
- **CORS**: Configurable allowed origins
- **Input Validation**: Pydantic schemas

### Data Security
- **PII Masking**: Sensitive fields hashed/rounded in logs
- **HTTPS**: TLS encryption (production)
- **Audit Trail**: Immutable decision logs

## Scalability

### Horizontal Scaling
- **Stateless API**: Can run multiple instances behind load balancer
- **Model Caching**: Models loaded once per instance
- **Async Processing**: FastAPI's async support

### Performance Optimization
- **Model Serving**: In-memory model loading
- **Preprocessor Caching**: Reuse fitted preprocessor
- **Batch Predictions**: Support batch API requests (future)

### Monitoring
- **Health Checks**: `/api/v1/health` endpoint
- **Metrics**: Prometheus integration (future)
- **Logging**: Structured logs for observability

## Deployment

### Docker Deployment
```bash
docker build -f infra/docker/Dockerfile -t kreditax:latest .
docker run -p 8000:8000 kreditax:latest
```

### Kubernetes Deployment (Future)
- Horizontal Pod Autoscaling
- Persistent Volume for audit logs
- ConfigMaps for environment configuration
- Secrets for sensitive data

## POJK Compliance Architecture

### Auditability
- Every prediction logged with unique ID
- Complete feature snapshot (masked)
- Model version tracking
- Export capability for regulators

### Explainability
- SHAP values for every prediction
- Local explanations (per-application)
- Global explanations (feature importance)
- Human-readable reasoning

### Data Privacy
- PII masking in logs
- No raw PII storage
- Statistical aggregation only
- GDPR-compliant data handling

## Technology Stack Summary

| Layer | Technology |
|-------|-----------|
| API | FastAPI, Uvicorn, Pydantic |
| ML | XGBoost, LightGBM, Scikit-learn, SHAP |
| Data | Pandas, NumPy |
| Storage | CSV (MVP), PostgreSQL (Future) |
| Deployment | Docker, Kubernetes (Future) |
| CI/CD | GitHub Actions |
| Testing | Pytest |

## Future Enhancements

1. **Authentication & Authorization**
   - OAuth2 / JWT implementation
   - Role-based access control (RBAC)

2. **Database Integration**
   - PostgreSQL for audit logs
   - Redis for caching
   - Model registry

3. **Monitoring & Observability**
   - Prometheus metrics
   - Grafana dashboards
   - Distributed tracing

4. **Advanced ML**
   - Online learning
   - A/B testing framework
   - Data drift detection
   - Bias monitoring

5. **React Dashboard**
   - Score distribution visualization
   - Feature importance plots
   - Audit log browser
   - Manual review interface
