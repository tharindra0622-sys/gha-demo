import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from calculator import add, subtract, multiply, divide, is_even


def test_add():
    assert add(2, 3) == 5


def test_subtract():
    assert subtract(5, 2) == 3


def test_multiply():
    assert multiply(4, 3) == 12


def test_divide():
    assert divide(10, 2) == 5


def test_divide_by_zero():
    with pytest.raises(ValueError):
        divide(1, 0)


def test_is_even():
    assert is_even(4) is True
    assert is_even(3) is False


def test_inject_failure():
    """Deliberately fails when INJECT_FAILURE=true is set, so you can
    trigger real, meaningful failing runs on demand — instead of only
    ever seeing successful 'Hello World' runs in your collected data."""
    if os.environ.get("INJECT_FAILURE", "false").lower() == "true":
        assert False, "Intentional failure triggered via INJECT_FAILURE"
    assert True
