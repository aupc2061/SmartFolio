from __future__ import annotations

import importlib
import logging
import sys
import contextlib
from typing import Any, Dict, Iterable, Optional, Type

import pathway as pw
pw.set_license_key('AE5CE3-7C24AE-8E3086-4D5E3E-CE4966-V3')

from .registry import ToolDefinition, iter_tools

logger = logging.getLogger(__name__)


class SmartFolioMCPServer:
    """Wrapper around Pathway's MCP server facilities (Updated for Pathway v0.27+)."""

    def __init__(self, app_id: str = "smartfolio-xai", host: str = "127.0.0.1", port: int = 9123, transport: str = "stdio"):
        self.app_id = app_id
        self.host = host
        self.port = port
        self.transport = transport

    def _load_pathway_mcp_primitives(self):
        if pw is None:
            raise ImportError(
                "Pathway is not installed. Install the official Pathway build to expose MCP tooling."
            )

        candidates = ("pathway.xpacks.llm.mcp_server",)
        last_err: Optional[Exception] = None
        for module_name in candidates:
            try:
                module = importlib.import_module(module_name)
                return module
            except Exception as exc:
                last_err = exc
        raise ImportError(
            f"Unable to import pathway MCP module. Ensure 'pathway[xpack-llm]' is installed."
        ) from last_err

    def _create_dynamic_schema(self, name: str, schema_def: Optional[Dict[str, Any]]) -> Type[pw.Schema]:
        """Dynamically creates a Pathway Schema class from a JSON-schema-like dict."""
        if not schema_def or "properties" not in schema_def:
            return type(f"{name}Schema", (pw.Schema,), {})

        annotations = {}
        for prop, details in schema_def.get("properties", {}).items():
            dtype = details.get("type", "string")
            if dtype == "integer":
                annotations[prop] = Optional[int]
            elif dtype == "number":
                annotations[prop] = Optional[float]
            elif dtype == "boolean":
                annotations[prop] = Optional[bool]
            else:
                annotations[prop] = Optional[str]
        
        return type(f"{name}Schema", (pw.Schema,), {"__annotations__": annotations})

    def serve(self) -> Any:
        mcp_module = self._load_pathway_mcp_primitives()
        
        PathwayMcp = getattr(mcp_module, "PathwayMcp")
        McpServer = getattr(mcp_module, "McpServer")
        McpServable = getattr(mcp_module, "McpServable")

        # 1. Helper UDF to pack columns into a dictionary
        #    Accepts **kwargs so it never fails signature validation.
        @pw.udf
        def pack_args(**kwargs) -> Dict[str, Any]:
            # Filter out the dummy trigger if present
            if '_trigger_id' in kwargs:
                del kwargs['_trigger_id']
            # Filter out None values so defaults can apply
            return {k: v for k, v in kwargs.items() if v is not None}

        class PythonFunctionTool(McpServable):
            def __init__(self, tool_def: ToolDefinition, schema_factory):
                self.tool_def = tool_def
                self.schema_cls = schema_factory(tool_def.name, tool_def.schema)

            def register_mcp(self, server):
                # 2. UDF to execute the tool logic using the packed dictionary
                @pw.udf
                def execute_tool(args: Any) -> Any:
                    try:
                        # Convert Pathway Json/wrapper to native dict
                        if not isinstance(args, dict):
                            # Strategy 1: Check for .value (Pathway Json wrapper often has this)
                            if hasattr(args, "value") and isinstance(args.value, dict):
                                args = args.value
                            # Strategy 2: Check for .as_dict()
                            elif hasattr(args, "as_dict"):
                                args = args.as_dict()
                            else:
                                # Strategy 3: Manual iteration
                                try:
                                    temp = {}
                                    # Iterating args usually yields keys
                                    for k in args:
                                        key_str = str(k)
                                        # Try accessing with original key, then string key
                                        try:
                                            val = args[k]
                                        except Exception:
                                            try:
                                                val = args[key_str]
                                            except Exception:
                                                continue
                                        temp[key_str] = val
                                    
                                    # Only replace args if we successfully extracted something
                                    if temp:
                                        args = temp
                                except Exception as e:
                                    sys.stderr.write(f"Warning: Failed to convert args to dict: {e}\n")

                        # Redirect stdout to stderr to prevent breaking MCP JSON-RPC protocol
                        with contextlib.redirect_stdout(sys.stderr):
                            return self.tool_def.handler(args)
                    except Exception as e:
                        sys.stderr.write(f"Error executing {self.tool_def.name}: {e}\n")
                        return {"error": str(e)}

                # 3. Request handler: Table -> Table
                def request_handler(input_table: pw.Table) -> pw.Table:
                    # Get argument names from schema
                    schema_props = self.tool_def.schema.get("properties", {}) if self.tool_def.schema else {}
                    arg_names = list(schema_props.keys())
                    
                    # Prepare arguments for packing
                    # Always include ID as a dummy trigger to prevent "0 arguments" error
                    pack_inputs = {'_trigger_id': input_table.id}
                    for name in arg_names:
                        pack_inputs[name] = input_table[name]

                    # Step A: Pack columns into a single dict column
                    args_col = pack_args(**pack_inputs)

                    # Step B: Execute tool with that dict
                    return input_table.select(
                        result=execute_tool(args_col)
                    )

                # Attach docstring for description
                request_handler.__doc__ = self.tool_def.description

                # Register
                server.tool(
                    name=self.tool_def.name,
                    request_handler=request_handler,
                    schema=self.schema_cls,
                )

        servables = []
        for definition in iter_tools():
            tool_instance = PythonFunctionTool(definition, self._create_dynamic_schema)
            servables.append(tool_instance)

        sys.stderr.write(f"Starting Pathway MCP Server via {self.transport}...\n")
        
        if self.transport == "sse":
            sys.stderr.write("DEBUG: Initializing PathwayMcp with transport='streamable-http'\n")
            server = PathwayMcp(
                name=self.app_id,
                transport="streamable-http",
                host=self.host,
                port=self.port,
                serve=servables
            )
        else:
            sys.stderr.write("DEBUG: Initializing PathwayMcp with transport='stdio'\n")
            server = PathwayMcp(
                name=self.app_id,
                transport="stdio", 
                serve=servables
            )
        
        # Start the engine
        return pw.run()


def list_registered_tools() -> Iterable[Dict[str, Any]]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "schema": tool.schema,
        }
        for tool in iter_tools()
    ]