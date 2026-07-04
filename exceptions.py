class LocalAIError(Exception):
    """Base exception for LocalAI client."""
    pass


class ConnectionError(LocalAIError):
    """Raised when unable to connect to LocalAI server."""
    pass


class ModelNotFoundError(LocalAIError):
    """Raised when requested model is not available."""
    pass


class TimeoutError(LocalAIError):
    """Raised when request times out."""
    pass


class APIError(LocalAIError):
    """Raised when LocalAI returns an error response."""

    def __init__(self, message: str, status_code: int = None, response_body: str = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body
