from enum import Enum
from pydantic import BaseModel

class ErrorCode(str, Enum):
    RATE_LIMIT = "rate_limit"
    EMPTY_RESULTS = "empty_results"
    QUERY_TOO_LONG = "query_too_long"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"
    FETCH_FAILED= "fetch_failed"


class ToolError(BaseModel):
    error: bool = True
    code: ErrorCode
    message: str