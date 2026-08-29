"""Tests for ensemble helpers."""

from midas.ensemble import behavioral_key


def test_behavioral_key_collapses_float_jitter() -> None:
    a = [{"AAA": 0.30000001, "BBB": 0.1}, {"AAA": 0.25}]
    b = [{"AAA": 0.30000002, "BBB": 0.1}, {"AAA": 0.25}]
    assert behavioral_key(a) == behavioral_key(b)


def test_behavioral_key_distinguishes_real_differences() -> None:
    a = [{"AAA": 0.300, "BBB": 0.1}]
    b = [{"AAA": 0.301, "BBB": 0.1}]
    assert behavioral_key(a) != behavioral_key(b)


def test_behavioral_key_is_order_insensitive_within_a_bar() -> None:
    assert behavioral_key([{"AAA": 0.3, "BBB": 0.1}]) == behavioral_key([{"BBB": 0.1, "AAA": 0.3}])


def test_behavioral_key_is_hashable_and_bar_order_sensitive() -> None:
    a = [{"AAA": 0.3}, {"AAA": 0.2}]
    b = [{"AAA": 0.2}, {"AAA": 0.3}]
    assert hash(behavioral_key(a)) != hash(behavioral_key(b))
