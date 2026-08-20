"""
Tool registry.

A tool is a python function the agent can ask the harness to run. The `@tool`
decorator registers it, which both makes it callable and lists it in the
prompt's tool catalogue.

Contract: the first positional argument is the `ToolContext`; the rest come
from the model's `tool_args` by keyword and must all have defaults, so a
malformed call degrades rather than crashes. Return a `ToolResult` (or a
string, wrapped automatically) and never raise - an error message the model can
read is more useful than a traceback.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional


@dataclass
class ToolContext:
    """Everything a tool may need to know about the task it is serving."""

    input_step: str                 # absolute path to the customer's STEP file
    work_dir: str                   # scratch dir for this task
    request_text: str               # the customer's edit request
    input_report: Optional[dict] = None   # cached analysis of input_step
    last_output_step: Optional[str] = None  # newest successfully built STEP
    last_report: Optional[dict] = None      # analysis of last_output_step
    last_script: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """What a tool sends back to the agent."""

    text: str                                   # goes into the conversation
    images: List[str] = field(default_factory=list)   # image paths to attach
    ok: bool = True


@dataclass
class Tool:
    name: str
    description: str
    params: Dict[str, str]
    func: Callable[..., Any]
    # Tools flagged expensive are hidden when the step budget is nearly spent.
    expensive: bool = False


_REGISTRY: Dict[str, Tool] = {}


def tool(name: str, description: str, params: Optional[Dict[str, str]] = None,
         expensive: bool = False):
    """Decorator that registers a function as an agent-callable tool."""

    def decorator(func):
        if name in _REGISTRY:
            raise ValueError(f"Duplicate tool name: {name!r}")
        _REGISTRY[name] = Tool(
            name=name,
            description=description,
            params=params or {},
            func=func,
            expensive=expensive,
        )
        return func

    return decorator


def get_tool(name: str) -> Optional[Tool]:
    return _REGISTRY.get(name)


def iter_tools() -> Iterator[Tool]:
    """Registered tools, in registration order."""
    return iter(_REGISTRY.values())


def tool_names() -> List[str]:
    return list(_REGISTRY.keys())


def run_tool(name: str, ctx: ToolContext, args: Optional[dict] = None) -> ToolResult:
    """Dispatch a tool call from the agent. Never raises."""
    entry = get_tool(name)
    if entry is None:
        return ToolResult(
            text=(
                f"ERROR: no tool named {name!r}. "
                f"Available tools: {', '.join(tool_names())}."
            ),
            ok=False,
        )

    args = args or {}
    if not isinstance(args, dict):
        return ToolResult(
            text=f"ERROR: tool_args for {name!r} must be a JSON object, got {type(args).__name__}.",
            ok=False,
        )

    try:
        result = entry.func(ctx, **args)
    except TypeError as exc:
        return ToolResult(
            text=(
                f"ERROR calling {name}: {exc}. "
                f"Accepted arguments: {', '.join(entry.params) or 'none'}."
            ),
            ok=False,
        )
    except Exception as exc:  # noqa: BLE001 - a tool must never kill the run
        return ToolResult(text=f"ERROR inside {name}: {type(exc).__name__}: {exc}", ok=False)

    if isinstance(result, ToolResult):
        return result
    return ToolResult(text=str(result))
