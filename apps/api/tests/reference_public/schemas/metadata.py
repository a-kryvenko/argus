"""Optional response metadata."""
from pydantic import Field

def metadata_field():
    return Field(default=None, exclude_if=lambda value: value is None,
                 description="Model or variable information for this response.")
