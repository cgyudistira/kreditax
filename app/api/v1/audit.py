"""
Audit Log Endpoint

Retrieve audit logs for compliance and monitoring.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from pydantic import BaseModel
from app.services.audit_service import AuditLogger
from app.core.config import get_settings
import pandas as pd

router = APIRouter()


class AuditLogResponse(BaseModel):
    """Audit log response model"""
    total_records: int
    logs: list


@router.get("/audit-log", response_model=AuditLogResponse)
async def get_audit_logs(
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records")
):
    """
    Retrieve audit logs.
    
    Args:
        start_date: Filter logs from this date
        end_date: Filter logs until this date
        limit: Maximum number of records to return
        
    Returns:
        Audit logs
    """
    try:
        settings = get_settings()
        audit_logger = AuditLogger(settings.audit_log_path)
        
        # Get logs
        logs_df = audit_logger.get_logs(
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        
        # Convert to list of dicts
        logs = logs_df.to_dict('records') if not logs_df.empty else []
        
        return AuditLogResponse(
            total_records=len(logs),
            logs=logs
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve audit logs: {str(e)}")


@router.get("/audit-log/export")
async def export_audit_logs():
    """
    Export all audit logs as CSV.
    
    Returns:
        CSV file download
    """
    try:
        from fastapi.responses import FileResponse
        
        settings = get_settings()
        audit_logger = AuditLogger(settings.audit_log_path)
        
        # Export to temp file
        export_path = "data/audit_logs_export.csv"
        audit_logger.export_logs(export_path)
        
        return FileResponse(
            export_path,
            media_type='text/csv',
            filename='kreditax_audit_logs.csv'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export audit logs: {str(e)}")
