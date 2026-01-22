from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import os

from app.utils.file_manager import FileManager

router = APIRouter()
file_manager = FileManager()

@router.get("/download/{filename}")
async def download_docx(filename: str):
    outputs_dir = file_manager.get_outputs_dir()
    file_path = outputs_dir / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        str(file_path),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=filename
    )
