from __future__ import annotations


def call_if_main(name: str = "__main__", discard: bool | None = None):
    """Decorator to ensure a function will call when the script is run as main."""
    if name != "__main__":
        discard = False if discard is None else discard
        if discard:
            return lambda _: None
        else:
            return lambda f: f
    else:
        discard = True if discard is None else discard
        if discard:
            return lambda f: (f() or True) and None
        else:
            return lambda f: (f() and None) or f


def divide_even(a: int, b: int) -> int:
    """Divides two integers"""
    assert a % b == 0, f"{a = } must be divisible by {b = }"
    return a // b


def divide_up(a: int, b: int) -> int:
    """Divides two integers, rounding up"""
    return (a + b - 1) // b


def divide_down(a: int, b: int) -> int:
    """Divides two integers, rounding down"""
    return a // b


class Unset:
    pass


UNSET = Unset()
