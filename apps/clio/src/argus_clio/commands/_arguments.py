"""Shared command argument types."""
import argparse
from datetime import UTC, datetime


def utc_hour(value):
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if result.tzinfo is None:
            raise ValueError('timezone required')
        result = result.astimezone(UTC)
        if result.minute or result.second or result.microsecond:
            raise ValueError('whole UTC hour required')
        return result
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


