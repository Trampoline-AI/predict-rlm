"""Shared host-side serialization helpers."""

from __future__ import annotations

import dataclasses
from decimal import Decimal
from enum import Enum
from typing import Any


def to_plain_data(value: Any) -> Any:
    """Convert rich Python objects into plain data containers.

    This is for host-side normalization before interpreter injection or
    transport. Unknown objects are deliberately preserved so backend-specific
    literal/transport code can fail with a precise unsupported-type error.
    """
    if isinstance(value, Decimal):
        # JSON has no decimal type.  Preserve exact precision for evidence and
        # transport by using the same string representation as Pydantic JSON.
        return str(value)

    if _is_pydantic_v2_model(value):
        return to_plain_data(_model_dump(value))

    if _is_pydantic_v1_model(value):
        return to_plain_data(value.dict())

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_plain_data(dataclasses.asdict(value))

    if isinstance(value, Enum):
        return to_plain_data(value.value)

    if isinstance(value, dict):
        return {key: to_plain_data(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_plain_data(item) for item in value]

    if isinstance(value, set):
        try:
            items = sorted(value)
        except TypeError:
            items = list(value)
        return [to_plain_data(item) for item in items]

    return value


def _is_pydantic_v2_model(value: Any) -> bool:
    return hasattr(value, "model_dump")


def _is_pydantic_v1_model(value: Any) -> bool:
    return hasattr(value, "dict") and hasattr(value, "__fields__")


def _model_dump(value: Any) -> Any:
    try:
        # Evidence is recorded as JSON.  Pydantic's JSON mode converts rich
        # standard values such as Decimal and date without losing their stable
        # wire representation.
        return value.model_dump(mode="json")
    except TypeError:
        return value.model_dump()
