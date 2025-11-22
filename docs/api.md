# KreditaX API Documentation

## Base URL
```
http://localhost:8000
```

## Authentication
Currently no authentication required. For production, implement JWT-based auth.

## Endpoints

### 1. Health Check

**GET** `/api/v1/health`

Check API health status.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "service": "KreditaX Credit Scoring API",
  "version": "1.0.0"
}
```

---

### 2. Credit Score Prediction

**POST** `/api/v1/predict`

Predict credit default probability for a loan application.

**Request Body**:
```json
{
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
  },
  "request_id": "optional-request-id",
  "user_id": "optional-user-id"
}
```

**Response**:
```json
{
  "request_id": "generated-or-provided-uuid",
  "prediction_score": 0.25,
  "risk_category": "LOW",
  "decision": "APPROVE",
  "explanation": {
    "prediction_probability": 0.25,
    "base_value": 0.35,
    "top_features": [
      {
        "feature": "debt_service_ratio",
        "shap_value": -0.15,
        "feature_value": 0.28,
        "impact": "decreases_risk"
      },
      ...
    ],
    "explanation": "Credit Default Risk: LOW (25.0% probability)\n\nKey factors...",
    "risk_category": "LOW"
  }
}
```

**Field Descriptions**:
- `prediction_score`: Probability of default (0-1)
- `risk_category`: VERY_LOW | LOW | MEDIUM | HIGH | VERY_HIGH
- `decision`: APPROVE | REJECT (based on threshold)
- `explanation`: SHAP-based explanation (if enabled)

---

### 3. Audit Logs

**GET** `/api/v1/audit-log`

Retrieve audit logs for compliance.

**Query Parameters**:
- `start_date` (optional): ISO format date
- `end_date` (optional): ISO format date
- `limit` (optional): Max records (default: 100, max: 1000)

**Response**:
```json
{
  "total_records": 50,
  "logs": [
    {
      "request_id": "uuid",
      "timestamp": "2024-01-01T12:00:00",
      "model_version": "1.0.0",
      "prediction_score": 0.25,
      "risk_category": "LOW",
      "decision": "APPROVE",
      "explanation_summary": "Low risk due to...",
      "masked_features_hash": "abc123",
      "user_id": "user_123"
    },
    ...
  ]
}
```

---

### 4. Export Audit Logs

**GET** `/api/v1/audit-log/export`

Export all audit logs as CSV file.

**Response**: CSV file download

---

## Data Models

### Gender
- `MALE`
- `FEMALE`

### Education
- `SD` - Elementary School
- `SMP` - Junior High
- `SMA` - Senior High
- `D3` - Diploma 3
- `S1` - Bachelor
- `S2` - Master
- `S3` - Doctoral

### Housing Type
- `OWNED`
- `RENTED`
- `PARENTS`

### Employment Status
- `PERMANENT`
- `CONTRACT`
- `SELF_EMPLOYED`
- `UNEMPLOYED`

## Error Responses

All endpoints may return:

```json
{
  "detail": "Error message description"
}
```

**HTTP Status Codes**:
- `200`: Success
- `400`: Bad Request (invalid input)
- `500`: Internal Server Error

## Rate Limits

No rate limits currently implemented. For production, implement rate limiting.

## Examples

### cURL Example
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

### Python Example
```python
import requests

url = "http://localhost:8000/api/v1/predict"
data = {
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

response = requests.post(url, json=data)
print(response.json())
```

## Interactive Documentation

Visit `http://localhost:8000/docs` for Swagger UI interactive documentation.
