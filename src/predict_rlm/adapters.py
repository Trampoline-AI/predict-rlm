"""Compatibility imports for input and output adapter contracts."""

from .runtime import (
    FieldDescriptor,
    InputAdapter,
    OutputAdapter,
    OutputReservation,
    PreparedInput,
    resolve_input_adapter,
    resolve_output_adapter,
)

__all__ = [
    "FieldDescriptor",
    "InputAdapter",
    "OutputAdapter",
    "OutputReservation",
    "PreparedInput",
    "resolve_input_adapter",
    "resolve_output_adapter",
]
