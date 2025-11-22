import pandas as pd
import numpy as np
import random
import uuid
from src.schema import Gender, Education, HousingType, EmploymentStatus

def generate_synthetic_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    random.seed(seed)
    
    data = []
    
    for _ in range(n_samples):
        # Demographics
        age = np.random.randint(21, 60)
        gender = np.random.choice([g.value for g in Gender])
        marital_status = np.random.choice(["SINGLE", "MARRIED", "DIVORCED"])
        education = np.random.choice([e.value for e in Education], p=[0.1, 0.1, 0.4, 0.1, 0.2, 0.05, 0.05])
        housing_type = np.random.choice([h.value for h in HousingType])
        
        # Financials
        # Base income based on education roughly
        base_income_map = {
            "SD": 3000000, "SMP": 4000000, "SMA": 5000000, 
            "D3": 7000000, "S1": 10000000, "S2": 15000000, "S3": 20000000
        }
        annual_income = base_income_map[education] * np.random.uniform(0.8, 2.0) * 12
        employment_status = np.random.choice([e.value for e in EmploymentStatus], p=[0.6, 0.2, 0.15, 0.05])
        work_experience_years = np.random.randint(0, age - 18)
        
        # Credit Data
        existing_loans_count = np.random.poisson(1)
        total_existing_debt = existing_loans_count * np.random.uniform(1000000, 50000000)
        credit_card_utilization = np.random.beta(2, 5)
        past_delinquencies = np.random.poisson(0.2)
        
        # Loan Request
        loan_amount = np.random.uniform(10000000, 200000000)
        loan_term_months = np.random.choice([6, 12, 24, 36, 48, 60])
        
        # Logic for Default Probability (Signal Injection)
        # High debt, low income, unemployment, past delinquencies increase risk
        monthly_income = annual_income / 12
        debt_service_ratio = (total_existing_debt * 0.03 + loan_amount / loan_term_months) / monthly_income
        
        risk_score = 0
        risk_score += (debt_service_ratio > 0.4) * 2
        risk_score += (past_delinquencies > 0) * 3
        risk_score += (employment_status == "UNEMPLOYED") * 4
        risk_score += (credit_card_utilization > 0.7) * 2
        risk_score += (age < 25) * 1
        
        # Sigmoid probability
        prob_default = 1 / (1 + np.exp(-(risk_score - 3)))
        is_default = np.random.binomial(1, prob_default)
        
        record = {
            "application_id": str(uuid.uuid4()),
            "age": age,
            "gender": gender,
            "marital_status": marital_status,
            "education": education,
            "housing_type": housing_type,
            "annual_income": int(annual_income),
            "employment_status": employment_status,
            "work_experience_years": work_experience_years,
            "existing_loans_count": existing_loans_count,
            "total_existing_debt": int(total_existing_debt),
            "credit_card_utilization": round(credit_card_utilization, 2),
            "past_delinquencies": past_delinquencies,
            "loan_amount": int(loan_amount),
            "loan_term_months": loan_term_months,
            "is_default": is_default
        }
        data.append(record)
        
    return pd.DataFrame(data)

if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    df = generate_synthetic_data(2000)
    df.to_csv("data/sample_credit_data.csv", index=False)
    print("Sample data generated at data/sample_credit_data.csv")
