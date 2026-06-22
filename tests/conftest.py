import os

# Provide a dummy key so the Tavily client can be constructed at import time.
# All network calls are mocked in the tests, so the value is never used.
os.environ.setdefault("TAVILY_API_KEY", "test-key")
