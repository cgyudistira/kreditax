# KreditaX Dashboard Guide

## Overview

The KreditaX Dashboard is a professional Streamlit-based web interface that provides:
- Interactive credit application submission
- Real-time risk assessment with SHAP explanations
- Model performance visualization
- Audit log viewing and analysis
- System configuration

## Technology Choice: Streamlit vs Gradio

**Why Streamlit?**

✅ **Chosen: Streamlit**
- More professional and polished UI
- Better layout control and customization
- Excellent data visualization support (Plotly integration)
- Superior state management for complex workflows
- Better suited for enterprise dashboards
- Strong community and ecosystem

❌ **Not Chosen: Gradio**
- More limited for complex dashboards
- Better suited for simple ML model demos
- Less customization options

## Features

### 1. Home Dashboard
- System overview and key metrics
- Quick start guide
- Feature highlights
- Security and compliance status

### 2. New Application Form
- **Personal Information**: Age, gender, marital status, education, housing, employment
- **Financial Information**: Income, work experience, existing debts, loans
- **Credit Information**: Credit card utilization, past delinquencies
- **Loan Request**: Amount and term
- **Real-time Validation**: Form validation with Pydantic schema
- **Instant Results**: Immediate risk assessment after submission

### 3. Results Visualization
- **Risk Score**: Default probability percentage
- **Risk Category**: VERY_LOW, LOW, MEDIUM, HIGH, VERY_HIGH
- **Decision**: APPROVE or REJECT recommendation
- **SHAP Explanations**: 
  - Top contributing features
  - Feature impact visualization
  - Human-readable explanation text
  - Interactive charts

### 4. Performance Dashboard
- **Model Metrics**: AUC-ROC, PR-AUC, Precision, Recall, F1-Score
- **Visualizations**:
  - ROC Curve
  - Precision-Recall Curve
  - Calibration Plot
  - Feature Importance
  - Confusion Matrix

### 5. Audit Log Viewer
- **Filtering**: By decision, date range, limit
- **Summary Statistics**: 
  - Total decisions
  - Approval rate
  - Average risk score
  - Model version
- **Visualizations**:
  - Decision distribution (pie chart)
  - Risk category distribution (bar chart)
  - Time series analysis
- **Export**: Download audit logs as CSV

### 6. Settings
- **Model Configuration**: Risk threshold adjustment
- **Feature Toggles**: Enable/disable explainability and audit logging
- **API Configuration**: Backend endpoint settings
- **System Information**: Version, status, compliance indicators

## Installation

### Prerequisites
```bash
pip install streamlit plotly requests
```

### Start Dashboard
```bash
streamlit run dashboard/app.py
```

The dashboard will open automatically in your default browser at `http://localhost:8501`

## Usage

### Step 1: Start Backend API
Before using the dashboard, ensure the FastAPI backend is running:

```bash
uvicorn app.main:app --reload
```

### Step 2: Launch Dashboard
```bash
streamlit run dashboard/app.py
```

### Step 3: Navigate
Use the sidebar to navigate between pages:
- 🏠 Home
- 📝 New Application
- 📊 Performance
- 📋 Audit Logs
- ⚙️ Settings

### Step 4: Submit Application
1. Go to "New Application" page
2. Fill in all required fields
3. Click "Assess Credit Risk"
4. View results and SHAP explanations

## Configuration

### API Endpoint
Update the API base URL in the Settings page or modify `API_BASE_URL` in `dashboard/app.py`:

```python
API_BASE_URL = "http://localhost:8000/api/v1"
```

### Custom Styling
The dashboard uses custom CSS for professional appearance. Colors and gradients can be customized in the `st.markdown()` CSS section.

### Risk Thresholds
Configure risk categories in the Settings page:
- Default threshold: 0.5 (50%)
- Adjustable from 0.0 to 1.0

## Architecture

```
Dashboard (Streamlit) → HTTP Requests → FastAPI Backend → ML Models
         ↓
    User Interface
         ↓
    Plotly Charts
         ↓
   SHAP Visualizations
```

## Key Components

### 1. Application Form (`page == "📝 New Application"`)
- Streamlit forms with validation
- Number inputs, select boxes, sliders
- Submit button triggers API call
- Results displayed with metrics and charts

### 2. Visualization (`plotly.express` & `plotly.graph_objects`)
- Interactive charts (zoom, pan, hover)
- Color-coded risk indicators
- SHAP feature importance bars
- Time series plots

### 3. API Integration
```python
def call_predict_api(application_data):
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json={"application": application_data}
    )
    return response.json()
```

### 4. State Management
- Streamlit session state for persistence
- Form state management
- Page navigation

## Screenshots

### Home Page
- Key metrics dashboard
- Quick start guide
- Feature highlights

### Application Form
- Multi-column layout
- Organized sections
- Clear labels and help text

### Results with SHAP
- Risk score display
- Decision badge
- SHAP horizontal bar chart
- Explanation text

### Performance Dashboard
- Tabbed interface
- High-quality plots
- Metrics at a glance

### Audit Logs
- Filterable data table
- Summary statistics
- Distribution charts
- CSV export

## Best Practices

### 1. User Experience
- Clear navigation with sidebar
- Consistent color scheme
- Helpful tooltips and info boxes
- Loading spinners for async operations

### 2. Data Validation
- Form validation before API calls
- Error handling for API failures
- User-friendly error messages

### 3. Performance
- Caching API responses (with `@st.cache_data`)
- Lazy loading of visualizations
- Optimized data transfer

### 4. Security
- No sensitive data stored in UI
- API authentication (future enhancement)
- PII masking in displayed logs

## Troubleshooting

### Dashboard doesn't load
- Check if port 8501 is available
- Try: `streamlit run dashboard/app.py --server.port 8502`

### API connection failed
- Ensure FastAPI backend is running on port 8000
- Check API_BASE_URL in settings
- Verify network connectivity

### Charts not displaying
- Install Plotly: `pip install plotly`
- Clear Streamlit cache: `streamlit cache clear`

### "Module not found" error
- Install dependencies: `pip install -r requirements.txt`
- Activate virtual environment

## Future Enhancements

### Phase 1 (Planned)
- [ ] User authentication and login
- [ ] Role-based access control
- [ ] Multi-language support (Indonesian/English)
- [ ] Dark mode toggle

### Phase 2 (Planned)
- [ ] Batch processing interface
- [ ] Advanced filtering and search
- [ ] Custom report generation
- [ ] Email notifications

### Phase 3 (Planned)
- [ ] Real-time model monitoring
- [ ] A/B testing dashboard
- [ ] Bias detection visualization
- [ ] Data drift alerts

## Support

For issues or questions:
- Check API status at `/api/v1/health`
- Review Streamlit logs in terminal
- Consult main documentation in `docs/`

## License

Proprietary - KreditaX
