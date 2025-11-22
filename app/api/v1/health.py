"""
Health Check Endpoint

Simple health check for monitoring and load balancers.
"""

from fastapi import APIRouter
from datetime import datetime

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Status information
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "KreditaX Credit Scoring API",
        "version": "1.0.0"
    }


@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "KreditaX Credit Scoring API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }
