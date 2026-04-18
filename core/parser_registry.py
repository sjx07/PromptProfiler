"""Global parser registry — eager autoload of all task parser modules.

Each task's parsers.py registers parsers via @register_parser into a local
PARSER_REGISTRY dict.  autoload_parsers() discovers and imports all
``tasks.<name>.parsers`` modules at package init time,
populating GLOBAL_PARSER_REGISTRY keyed by module path.

This replaces the per-call lazy importlib.import_module() path in
BaseTask._get_parser_registry().

Usage (automatic — called from prompt_profiler/__init__.py):
    from core.parser_registry import autoload_parsers
    autoload_parsers()

Direct lookup (from task.py):
    from core.parser_registry import GLOBAL_PARSER_REGISTRY
    registry = GLOBAL_PARSER_REGISTRY.get(module_path)
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# module_path (str) -> PARSER_REGISTRY dict from that module
# e.g. "tasks.wtq.parsers" -> {"code": <fn>, "answer": <fn>, ...}
GLOBAL_PARSER_REGISTRY: Dict[str, Dict[str, Callable[[str, Any], str]]] = {}

_autoloaded = False


def autoload_parsers() -> None:
    """Discover and import all ``tasks.<name>.parsers`` modules.

    Idempotent — calling multiple times has no effect after the first call.
    Missing parsers modules (tasks without one) are silently skipped.
    Import errors other than ModuleNotFoundError are logged as warnings.
    """
    global _autoloaded
    if _autoloaded:
        return
    _autoloaded = True

    try:
        import tasks as tasks_pkg
    except ImportError:
        logger.warning("autoload_parsers: could not import tasks")
        return

    for _, name, is_pkg in pkgutil.iter_modules(tasks_pkg.__path__):
        if not is_pkg:
            continue  # only recurse into sub-packages
        module_path = f"tasks.{name}.parsers"
        try:
            mod = importlib.import_module(module_path)
            registry = getattr(mod, "PARSER_REGISTRY", None)
            if registry is not None:
                GLOBAL_PARSER_REGISTRY[module_path] = registry
                logger.debug("autoload_parsers: loaded %s (%d parsers)", module_path, len(registry))
        except ModuleNotFoundError:
            pass  # task has no parsers.py — expected
        except Exception as exc:
            logger.warning("autoload_parsers: error loading %s: %s", module_path, exc)


def get_parser_registry(module_path: Optional[str]) -> Optional[Dict[str, Callable[[str, Any], str]]]:
    """Return the PARSER_REGISTRY for a given module path, or None.

    Looks up GLOBAL_PARSER_REGISTRY first.  Falls back to importlib if the
    module was not autoloaded (e.g. in tests that construct tasks before
    autoload_parsers() has run).
    """
    if module_path is None:
        return None

    # Fast path: already autoloaded
    if module_path in GLOBAL_PARSER_REGISTRY:
        return GLOBAL_PARSER_REGISTRY[module_path]

    # Fallback: lazy import (for test isolation / unusual import orders)
    try:
        mod = importlib.import_module(module_path)
        registry = getattr(mod, "PARSER_REGISTRY", None)
        if registry is not None:
            GLOBAL_PARSER_REGISTRY[module_path] = registry
        return registry
    except ImportError:
        return None
