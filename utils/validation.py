"""
Input validation utilities for uploaded files.
"""

import os
from werkzeug.utils import secure_filename
from config.settings import ALLOWED_EXTENSIONS, MAX_UPLOAD_BYTES


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_upload(file) -> tuple[bool, str]:
    """
    Validate uploaded file.
    
    Returns:
        (is_valid, error_message)
    """
    if not file:
        return False, "No file provided"
    
    if file.filename == "":
        return False, "Empty filename"
    
    if not allowed_file(file.filename):
        return False, f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    
    # Check file size (if possible)
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    
    if size > MAX_UPLOAD_BYTES:
        return False, f"File too large. Max size: {MAX_UPLOAD_BYTES // (1024*1024)} MB"
    
    return True, ""


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal."""
    return secure_filename(filename)
