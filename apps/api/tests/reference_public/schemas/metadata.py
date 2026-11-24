"""Optional descriptions must not change the data or erase meaningful nulls."""
from typing import Annotated

from fastapi import Query
from pydantic import Field

MetaQuery = Annotated[bool, Query(description="Include descriptive metadata in data.meta. Defaults to false; data values and structure are unchanged.")]


def metadata_field():
    return Field(default=None, exclude_if=lambda value: value is None,
                 description="Additional descriptions, included only with meta=true.")
