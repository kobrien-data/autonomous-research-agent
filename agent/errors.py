from enum import Enum
from pydantic import BaseModel

class ErrorCode(str, Enum):
    RATE_LIMIT = "rate_limit"
    EMPTY_RESULTS = "empty_results"
    QUERY_TOO_LONG = "query_too_long"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"
    FETCH_FAILED= "fetch_failed"
    CONNECTION_ERROR = "connection_error"
    FILE_NOT_FOUND = "file_not_found"
    SCANNED_PDF_ERROR = "scanned_pdf_error"
    PASSWORD_PROTECTED = "password_protected"


class ToolError(BaseModel):
    error: bool = True
    code: ErrorCode
    message: str

class ToolException(Exception):
    """Raised when a tool operation fails; carries a structured ToolError."""
    def __init__(self, error: ToolError):
        self.error = error
        super().__init__(error.message)
