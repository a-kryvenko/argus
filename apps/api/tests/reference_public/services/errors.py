"""Errors raised by the API forecast read adapter."""


class ArtifactNotReadyError(Exception):
    """A forecast release is unavailable or cannot be rendered by the API."""
