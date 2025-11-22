from pydantic import BaseModel, Field, validator
from typing import Optional
from enum import Enum

class Gender(str, Enum):
    MALE = "MALE"
    FEMALE = "FEMALE"

class Education(str, Enum):
    SD = "SD"
    SMP = "SMP"
    SMA = "SMA"
    D3 = "D3"
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"

class HousingType(str, Enum):
    OWNED = "OWNED"
    RENTED = "RENTED"
    PARENTS = "PARENTS"

class EmploymentStatus(str, Enum):
    PERMANENT = "PERMANENT"
    CONTRACT = "CONTRACT"
    SELF_EMPLOYED = "SELF_EMPLOYED"
    UNEMPLOYED = "UNEMPLOYED"

class CreditApplication(BaseModel):
    application_id: str = Field(..., description="Unique identifier for the application")
    
    # Demographics
    age: int = Field(..., ge=18, le=70, description="Applicant age")
    gender: Gender
    marital_status: str
    education: Education
    housing_type: HousingType
    
    # Financials
    annual_income: float = Field(..., gt=0, description="Annual income in IDR")
    employment_status: EmploymentStatus
    work_experience_years: int = Field(..., ge=0)
    
    # Credit Data
    existing_loans_count: int = Field(..., ge=0)
    total_existing_debt: float = Field(..., ge=0)
    credit_card_utilization: float = Field(..., ge=0.0, le=1.0)
    past_delinquencies: int = Field(..., ge=0)
    
    # Loan Request
    loan_amount: float = Field(..., gt=0)
    loan_term_months: int = Field(..., ge=1, le=60)
    
    # Target (Optional for inference)
    is_default: Optional[int] = Field(None, description="0: Good, 1: Default")

    class Config:
        schema_extra = {
            "example": {
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
