"""
Credit Application Schema

Pydantic models for credit application data validation.
All field descriptions follow Indonesian banking standards and POJK compliance.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class Gender(str, Enum):
    """Gender enumeration"""
    MALE = "MALE"
    FEMALE = "FEMALE"


class Education(str, Enum):
    """Education level enumeration (Indonesian system)"""
    SD = "SD"      # Elementary School
    SMP = "SMP"    # Junior High School
    SMA = "SMA"    # Senior High School
    D3 = "D3"      # Diploma 3
    S1 = "S1"      # Bachelor's Degree
    S2 = "S2"      # Master's Degree
    S3 = "S3"      # Doctoral Degree


class HousingType(str, Enum):
    """Housing ownership type"""
    OWNED = "OWNED"
    RENTED = "RENTED"
    PARENTS = "PARENTS"


class EmploymentStatus(str, Enum):
    """Employment status enumeration"""
    PERMANENT = "PERMANENT"
    CONTRACT = "CONTRACT"
    SELF_EMPLOYED = "SELF_EMPLOYED"
    UNEMPLOYED = "UNEMPLOYED"


class CreditApplication(BaseModel):
    """
    Credit Application Data Model
    
    This model represents a complete credit application with all required
    fields for risk assessment and POJK compliance.
    """
    
    application_id: str = Field(..., description="Unique identifier for the application")
    
    # Demographics
    age: int = Field(..., ge=18, le=70, description="Applicant age")
    gender: Gender
    marital_status: str = Field(..., description="Marital status (SINGLE, MARRIED, DIVORCED)")
    education: Education
    housing_type: HousingType
    
    # Financials
    annual_income: float = Field(..., gt=0, description="Annual income in IDR")
    employment_status: EmploymentStatus
    work_experience_years: int = Field(..., ge=0, description="Years of work experience")
    
    # Credit Data
    existing_loans_count: int = Field(..., ge=0, description="Number of existing loans")
    total_existing_debt: float = Field(..., ge=0, description="Total existing debt in IDR")
    credit_card_utilization: float = Field(..., ge=0.0, le=1.0, description="Credit card utilization ratio")
    past_delinquencies: int = Field(..., ge=0, description="Number of past payment delinquencies")
    
    # Loan Request
    loan_amount: float = Field(..., gt=0, description="Requested loan amount in IDR")
    loan_term_months: int = Field(..., ge=1, le=60, description="Loan term in months")
    
    # Target (Optional for inference)
    is_default: Optional[int] = Field(None, description="0: Good, 1: Default")

    class Config:
        """Pydantic configuration"""
        json_schema_extra = {
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
