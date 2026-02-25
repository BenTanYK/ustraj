import importlib


def test_ustraj():
    assert importlib.import_module("ustraj") is not None
