# KreditaX Dashboard

Professional Streamlit-based UI for interactive credit scoring.

## Quick Start

```bash
# Start backend API first
uvicorn app.main:app --reload

# In a new terminal, start dashboard
streamlit run dashboard/app.py
```

Access at: **http://localhost:8501**

## Features

- 📝 **Application Form**: Interactive credit application submission
- 📊 **Risk Assessment**: Real-time predictions with SHAP explanations
- 📈 **Performance Dashboard**: Model metrics and visualizations
- 📋 **Audit Logs**: Decision tracking and analysis
- ⚙️ **Settings**: System configuration

## Pages

### 1. Home (🏠)
- System overview
- Key metrics
- Quick start guide

### 2. New Application (📝)
- Personal information form
- Financial details
- Credit history
- Loan request
- Real-time risk assessment

### 3. Performance (📊)
- Model metrics (AUC, PR-AUC, etc.)
- ROC & PR curves
- Calibration plots
- Feature importance

### 4. Audit Logs (📋)
- Decision history
- Filter and search
- Summary statistics
- CSV export

### 5. Settings (⚙️)
- Risk threshold configuration
- Feature toggles
- API endpoint settings

## Documentation

See [docs/DASHBOARD.md](../docs/DASHBOARD.md) for complete documentation.

## Technology

- **Framework**: Streamlit 1.28+
- **Visualization**: Plotly 5.17+
- **API Client**: Requests 2.31+

## Requirements

Installed automatically with `pip install -r requirements.txt`
