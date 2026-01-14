"""
Tradetron integration package for paper trading commodities.

This package contains all Tradetron-related code:
- TradetronClient: API client for sending signals to Tradetron
- Test scripts and documentation
"""

from .tradetron_client import (
    TradetronClient,
    TradetronConfig,
    TradetronAuthError,
)

__all__ = [
    "TradetronClient",
    "TradetronConfig",
    "TradetronAuthError",
]
