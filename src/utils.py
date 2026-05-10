"""Common utility functions for the AegisHarness Demo project."""
from typing import Any


def validate_input(value: Any, expected_type: type, name: str = "value") -> Any:
    """Validate that a value is of the expected type.
    
    Args:
        value: The value to validate.
        expected_type: The expected Python type.
        name: Name of the parameter for error messages.
        
    Returns:
        The validated value.
        
    Raises:
        TypeError: If value is not of expected_type.
        ValueError: If value is None when not expected.
    """
    if value is None:
        raise ValueError(f"{name} must not be None")
    if not isinstance(value, expected_type):
        raise TypeError(f"{name} must be {expected_type.__name__}, got {type(value).__name__}")
    return value
