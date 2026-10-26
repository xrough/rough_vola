"""Notebook-session setup remains informative when optional imports are broken."""

import builtins

from rvlab import _probe


def test_optional_probe_reports_non_import_failure(monkeypatch):
    original_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "broken_optional":
            raise RuntimeError("missing native runtime")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    assert _probe("broken_optional") == "UNAVAILABLE (RuntimeError)"
