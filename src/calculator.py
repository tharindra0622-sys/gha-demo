"""A tiny calculator module — just enough real code to give CI something
meaningful to lint, test, and build."""


def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def multiply(a, b):
    return a * b


def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def is_even(n):
    return n % 2 == 0
