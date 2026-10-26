"""
rvlab.compat
============
The selected ``roughvol`` components ship in this distribution. No sibling
checkout is discovered or placed on sys.path. Legacy flat research modules
remain an explicit, optional local input through RVLAB_RESEARCH_SHARED_DIR.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

from . import config

HAS_ROUGHVOL: bool | None = None          # None = not yet probed
HAS_RESEARCH_SHARED: bool | None = None

_ROUGHVOL_PATH: Path | None = None
_RESEARCH_SHARED_PATH: Path | None = None


def _prepend(path: Path) -> None:
    """Put `path` at the front of sys.path exactly once."""
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)


def ensure_roughvol(quiet: bool = True) -> bool:
    """Check the bundled namespace without injecting external source paths."""
    global HAS_ROUGHVOL, _ROUGHVOL_PATH
    if HAS_ROUGHVOL:
        return True

    try:                                    # already importable (e.g. pip -e)?
        mod = importlib.import_module("roughvol")
        importlib.import_module("roughvol.types")
        HAS_ROUGHVOL = True
        _ROUGHVOL_PATH = Path(mod.__file__).resolve().parent.parent if mod.__file__ else None
        return True
    except ImportError:
        pass

    HAS_ROUGHVOL = False
    return False


def ensure_research_shared(quiet: bool = True) -> bool:
    """Make the flat modules in demo/python/research/shared/ importable.

    They import each other by bare name (`from smile_pipeline import ...`), so
    the directory itself has to be on sys.path, not its parent.
    """
    global HAS_RESEARCH_SHARED, _RESEARCH_SHARED_PATH
    if HAS_RESEARCH_SHARED:
        return True

    shared = config.RESEARCH_SHARED_DIR
    if not (shared / "smile_pipeline.py").exists():
        HAS_RESEARCH_SHARED = False
        return False

    _prepend(shared)
    try:
        importlib.import_module("smile_pipeline")
    except ImportError as exc:              # e.g. databento not installed
        if not quiet:
            print(f"[rvlab.compat] research/shared failed to import: {exc}")
        HAS_RESEARCH_SHARED = False
        return False

    HAS_RESEARCH_SHARED, _RESEARCH_SHARED_PATH = True, shared
    return True


def roughvol_module(name: str) -> ModuleType:
    """Import a roughvol submodule, raising a directive error if unavailable."""
    if not ensure_roughvol():
        raise ImportError(
            "Bundled roughvol components are unavailable; install the complete rough-vola distribution."
        )
    return importlib.import_module(name)


def research_module(name: str) -> ModuleType:
    """Import a demo/python/research/shared module, raising if unavailable."""
    if not ensure_research_shared():
        raise ImportError(
            f"demo/python/research/shared/{name}.py is not importable. "
            "It needs the local research tree and the `databento` package."
        )
    return importlib.import_module(name)


def paths() -> dict[str, str | None]:
    """Where compat actually found things — printed by the notebook env report."""
    return {
        "roughvol": str(_ROUGHVOL_PATH) if _ROUGHVOL_PATH else None,
        "research_shared": str(_RESEARCH_SHARED_PATH) if _RESEARCH_SHARED_PATH else None,
    }


__all__ = [
    "ensure_roughvol", "ensure_research_shared", "roughvol_module",
    "research_module", "paths", "HAS_ROUGHVOL", "HAS_RESEARCH_SHARED",
]
