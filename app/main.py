"""
KreditaX FastAPI Application

Main application entry point for credit scoring API.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import get_settings
from app.api.v1 import health, predict, audit

# Get settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title="KreditaX Credit Scoring API",
    description="AI-powered credit scoring engine with POJK compliance",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix=settings.api_v1_prefix, tags=["health"])
app.include_router(predict.router, prefix=settings.api_v1_prefix, tags=["prediction"])
app.include_router(audit.router, prefix=settings.api_v1_prefix, tags=["audit"])


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("="*50)
    print("🚀 Kr editaX API Starting...")
    print("="*50)
    print(f"Version: {settings.app_version}")
    print(f"API Prefix: {settings.api_v1_prefix}")
    print(f"Model: {settings.model_path}")
    print(f"Preprocessor: {settings.preprocessor_path}")
    print(f"Audit Logging: {'Enabled' if settings.enable_audit_logging else 'Disabled'}")
    print(f"Explainability: {'Enabled' if settings.enable_explainability else 'Disabled'}")
    print("="*50)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("👋 KreditaX API shutting down...")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
