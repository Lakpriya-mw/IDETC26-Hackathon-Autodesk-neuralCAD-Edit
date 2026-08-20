"""
Tool package.

Every `*.py` file here (except `base.py` and files starting with `_`) is
imported on load, so any `@tool`-decorated function inside it registers itself.
"""

import importlib
import os
import pkgutil

from harness.tools.base import (  # noqa: F401  (re-exported for convenience)
    Tool,
    ToolContext,
    ToolResult,
    get_tool,
    iter_tools,
    run_tool,
    tool,
    tool_names,
)

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_SKIP = {"base"}


def _discover():
    for module_info in pkgutil.iter_modules([_PKG_DIR]):
        name = module_info.name
        if name in _SKIP or name.startswith("_"):
            continue
        importlib.import_module(f"{__name__}.{name}")


_discover()
