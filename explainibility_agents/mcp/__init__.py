from __future__ import annotations

from .config import XAIRequest
from .mcp_server import SmartFolioMCPServer, list_registered_tools

# Importing these modules registers their MCP handlers with the shared registry.
from . import explain_tree as _explain_tree  # noqa: F401
from . import run_trading_agents as _trading  # noqa: F401
from . import orchestrator_xai as _orchestrator  # noqa: F401

__all__ = [
	"SmartFolioMCPServer",
	"XAIRequest",
	"list_registered_tools",
]
