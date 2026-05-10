"""Tests for utility functions."""
import pytest
from src.utils import validate_input


def test_validate_input_correct_type():
    assert validate_input(42, int) == 42


def test_validate_input_wrong_type():
    with pytest.raises(TypeError):
        validate_input("hello", int)


def test_validate_input_none():
    with pytest.raises(ValueError):
        validate_input(None, int)
