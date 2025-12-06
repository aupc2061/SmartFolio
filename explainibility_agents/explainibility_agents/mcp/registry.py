from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional


@dataclass
class ToolDefinition:
    name: str
    description: str
    handler: Callable[[Dict[str, Any]], Dict[str, Any]]
    schema: Optional[Dict[str, Any]] = None


_TOOL_REGISTRY: Dict[str, ToolDefinition] = {}


def register_mcp_tool(
    *,
    name: str,
    description: str,
    schema: Optional[Dict[str, Any]] = None,
) -> Callable[[Callable[[Dict[str, Any]], Dict[str, Any]]], Callable[[Dict[str, Any]], Dict[str, Any]]]:
    """Decorator used by tool modules to self-register MCP handlers."""

    def decorator(func: Callable[[Dict[str, Any]], Dict[str, Any]]):
        _TOOL_REGISTRY[name] = ToolDefinition(name=name, description=description, handler=func, schema=schema)
        return func

    return decorator


def iter_tools() -> Iterable[ToolDefinition]:
    return _TOOL_REGISTRY.values()
