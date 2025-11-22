# KreditaX Quick Start Guide

## Prerequisites
- Python 3.10+
- pip

## Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Generate Sample Data (Demo)
```bash
python -m scripts.generate_data
```

Output: `data/demo/sample_credit_data.csv`

### 2. Train Model
```bash
python -m scripts.train_model
```

Options:
- `--model xgboost` or `--model lightgbm`
- `--test-size 0.2`
- `--cv-folds 5`

Output:
- Model: `ml/artifacts/xgboost_model_latest.joblib`
- Preprocessor: `ml/artifacts/preprocessor.joblib`
- Evaluation plots: `ml/experiments/`

### 3. Start API Server
```bash
uvicorn app.main:app --reload
```

Access:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health: http://localhost:8000/api/v1/health

### 4. Make Prediction

**Using cURL**:
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "application": {
      "application_id": "APP-001",
      "age": 30,
      "gender": "MALE",
      "marital_status": "MARRIED",
      "education": "S1",
      "housing_type": "OWNED",
      "annual_income": 120000000,
      "employment_status": "PERMANENT",
      "work_experience_years": 5,
      "existing_loans_count": 1,
      "total_existing_debt": 5000000,
      "credit_card_utilization": 0.3,
      "past_delinquencies": 0,
      "loan_amount": 50000000,
      "loan_term_months": 12
    }
  }'
```

**Using Python**:
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/predict",
    json={
        "application": {
            "application_id": "APP-001",
            "age": 30,
            "gender": "MALE",
            "marital_status": "MARRIED",
            "education": "S1",
            "housing_type": "OWNED",
            "annual_income": 120000000,
            "employment_status": "PERMANENT",
            "work_experience_years": 5,
            "existing_loans_count": 1,
            "total_existing_debt": 5000000,
            "credit_card_utilization": 0.3,
            "past_delinquencies": 0,
            "loan_amount": 50000000,
            "loan_term_months": 12
        }
    }
)

result = response.json()
print(f"Risk Score: {result['prediction_score']:.2%}")
print(f"Decision: {result['decision']}")
print(f"Risk Category: {result['risk_category']}")
```

### 5. View Audit Logs
```bash
curl "http://localhost:8000/api/v1/audit-log?limit=10"
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=app --cov=ml

# Specific tests
pytest tests/ml/test_preprocessing.py -v
pytest tests/app/test_api.py -v
```

## Docker

```bash
# Build
docker build -f infra/docker/Dockerfile -t kreditax:latest .

# Run
docker run -p 8000:8000 kreditax:latest

# Test
curl http://localhost:8000/api/v1/health
```

## Configuration

Copy `.env.example` to `.env` and customize:

```env
KREDITAX_MODEL_PATH=ml/artifacts/xgboost_model_latest.joblib
KREDITAX_PREPROCESSOR_PATH=ml/artifacts/preprocessor.joblib
KREDITAX_DEFAULT_RISK_THRESHOLD=0.5
KREDITAX_ENABLE_EXPLAINABILITY=true
KREDITAX_ENABLE_AUDIT_LOGGING=true
```

## Common Issues

### Model not found
**Error**: `Model not found at ml/artifacts/...`
**Solution**: Run `python -m scripts.train_model` first

### Import errors
**Error**: `ModuleNotFoundError: No module named 'app'`
**Solution**: Run from project root, use `python -m` syntax

### Port already in use
**Error**: `Address already in use`
**Solution**: Change port with `--port 8001` or stop other service

## Documentation

- **API**: `docs/api.md`
- **Architecture**: `docs/architecture.md`
- **POJK Compliance**: `docs/pojk-compliance.md`
- **Full Guide**: `README.md`

## Support

Check Swagger UI at `/docs` for interactive API documentation.
