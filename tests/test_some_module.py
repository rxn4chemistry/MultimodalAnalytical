"""Some module tests."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""
from mmbart.some_module import dummy_function


def test_dummy_function() -> None:
    """Function to test dummy module."""
    assert dummy_function(10) == 20
