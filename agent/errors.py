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
    CODE_TOO_LONG = "code_too_long"
    BLOCKED_IMPORT = "blocked_import"
    EXECUTION_FAILED = "execution_failed"
    INVALID_INPUT = "invalid_input"
    AUTH_ERROR = "auth_error"


# Transient failures worth retrying; everything else (bad input, missing
# files, auth) will fail the same way on a second attempt.
RETRYABLE_CODES = frozenset({
    ErrorCode.RATE_LIMIT,
    ErrorCode.TIMEOUT,
    ErrorCode.CONNECTION_ERROR,
})


class ToolError(BaseModel):
    error: bool = True
    code: ErrorCode
    message: str
    tool_name: str = ""
    retryable: bool = False

    def model_post_init(self, __context) -> None:
        self.retryable = self.code in RETRYABLE_CODES

class ToolException(Exception):
    """Raised when a tool operation fails; carries a structured ToolError."""
    def __init__(self, error: ToolError):
        self.error = error
        super().__init__(error.message)
