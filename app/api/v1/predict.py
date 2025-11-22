"""
Prediction Endpoint

Credit scoring prediction API with POJK compliance.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from app.schemas.credit_application import CreditApplication
from app.services.model_service import get_model_service, ModelService
from app.services.audit_service import AuditLogger
from app.services.explain_service import SHAPExplainer
from app.core.config import get_settings
import uuid

router = APIRouter()


class PredictionRequest(BaseModel):
    """Prediction request model"""
    application: CreditApplication
    request_id: Optional[str] = None
    user_id: Optional[str] = None


class PredictionResponse(BaseModel):
    """Prediction response model"""
    request_id: str
    prediction_score: float
    risk_category: str
    decision: str
    explanation: Optional[dict] = None


@router.post("/predict", response_model=PredictionResponse)
async def predict_credit_score(
    request: PredictionRequest,
    model_service: ModelService = Depends(get_model_service)
):
    """
    Predict credit default probability.
    
    Args:
        request: Prediction request with application data
        
    Returns:
        Prediction response with score, risk category, and explanation
    """
    try:
        # Generate request ID if not provided
        request_id = request.request_id or str(uuid.uuid4())
        
        # Convert Pydantic model to dict
        features = request.application.model_dump()
        
        # Make prediction
        prediction_score, risk_category = model_service.predict(features)
        
        # Determine decision
        settings = get_settings()
        threshold = settings.default_risk_threshold
        decision = "REJECT" if prediction_score > threshold else "APPROVE"
        
        # Generate explanation if enabled
        explanation = None
        if settings.enable_explainability:
            try:
                # Get processed features
                X_processed = model_service.preprocess(features)
                
                # Initialize explainer
                explainer = SHAPExplainer(model_service.model)
                
                # Get explanation
                explanation = explainer.explain_prediction(
                    X_processed[0],
                    feature_names=model_service.get_feature_names(),
                    top_k=5
                )
            except Exception as e:
                print(f"Warning: Could not generate explanation: {e}")
                explanation = {"error": "Explanation generation failed"}
        
        # Log to audit system
        if settings.enable_audit_logging:
            try:
                audit_logger = AuditLogger(settings.audit_log_path)
                audit_logger.log_prediction(
                    request_id=request_id,
                    features=features,
                    prediction_score=prediction_score,
                    risk_category=risk_category,
                    explanation=explanation or {},
                    user_id=request.user_id
                )
            except Exception as e:
                print(f"Warning: Audit logging failed: {e}")
        
        return PredictionResponse(
            request_id=request_id,
            prediction_score=prediction_score,
            risk_category=risk_category,
            decision=decision,
            explanation=explanation
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
