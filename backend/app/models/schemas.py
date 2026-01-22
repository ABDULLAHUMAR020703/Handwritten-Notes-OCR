from pydantic import BaseModel
from typing import Optional

class UploadResponse(BaseModel):
    status: str
    output_path: Optional[str] = None
    message: Optional[str] = None

class ErrorResponse(BaseModel):
    detail: str
