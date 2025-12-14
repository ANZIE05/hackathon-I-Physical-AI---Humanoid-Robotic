from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import time
from typing import Optional


class SecurityMiddleware:
    """
    Security middleware for the RAG API
    """
    def __init__(self):
        self.api_key = os.getenv("API_KEY")
        self.rate_limit_window = 60  # 60 seconds
        self.max_requests_per_window = 100
        self.request_times = {}

    def check_rate_limit(self, client_ip: str) -> bool:
        """
        Basic rate limiting implementation
        """
        current_time = time.time()

        if client_ip not in self.request_times:
            self.request_times[client_ip] = []

        # Remove requests older than the rate limit window
        self.request_times[client_ip] = [
            req_time for req_time in self.request_times[client_ip]
            if current_time - req_time < self.rate_limit_window
        ]

        # Check if the client has exceeded the rate limit
        if len(self.request_times[client_ip]) >= self.max_requests_per_window:
            return False

        # Add the current request time
        self.request_times[client_ip].append(current_time)
        return True

    def validate_api_key(self, credentials: HTTPAuthorizationCredentials) -> bool:
        """
        Validate the API key
        """
        if not self.api_key:
            # If no API key is set, allow all requests (development mode)
            return True

        if not credentials or not credentials.credentials:
            return False

        return credentials.credentials == self.api_key


# Initialize security middleware
security_middleware = SecurityMiddleware()


def require_api_key(request: Request) -> bool:
    """
    Dependency to require API key authentication
    """
    auth_header = request.headers.get("Authorization")

    if not auth_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is required"
        )

    # Basic validation - in production, use proper JWT or API key validation
    if not security_middleware.validate_api_key(None):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    # Check rate limit
    client_ip = request.client.host
    if not security_middleware.check_rate_limit(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )

    return True