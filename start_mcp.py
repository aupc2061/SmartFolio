import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Workaround for fastmcp compatibility issues with Pathway MCP server
# 1. Fix NotSet import issue
try:
    import fastmcp.utilities.types as fastmcp_types
    if not hasattr(fastmcp_types, 'NotSet'):
        class NotSet:
            """Placeholder for NotSet when fastmcp version doesn't include it."""
            def __repr__(self):
                return "NotSet"
            def __bool__(self):
                return False
        fastmcp_types.NotSet = NotSet
        setattr(fastmcp_types, 'NotSet', NotSet)
except (ImportError, AttributeError):
    import types
    fastmcp_types = types.ModuleType('fastmcp.utilities.types')
    class NotSet:
        """Placeholder for NotSet when fastmcp is not available."""
        def __repr__(self):
            return "NotSet"
        def __bool__(self):
            return False
    fastmcp_types.NotSet = NotSet
    sys.modules['fastmcp.utilities.types'] = fastmcp_types

# 2. Fix FastMCP.tool() compatibility issues
# Pathway's McpServer.tool() accepts many parameters (title, output_schema, etc.)
# but FastMCP.tool() only accepts: name, description, tags (all as keyword args)
# Pathway calls: self._fastmcp.tool(name=..., title=..., output_schema=..., ...)
# We need to filter out unsupported arguments and map 'title' to 'description' if needed
try:
    from fastmcp import FastMCP
    import functools
    import types
    
    # Get the original tool method (it's a descriptor/decorator)
    if hasattr(FastMCP, 'tool'):
        original_tool = FastMCP.tool
        
        @functools.wraps(original_tool)
        def patched_tool(self, *args, **kwargs):
            """Wrapper that filters out unsupported FastMCP arguments.
            
            Pathway calls: FastMCP.tool(handler_function, name='route', title=..., ...)
            But FastMCP.tool() signature is: tool(self, name=None, description=None, tags=None)
            
            The first positional arg (after self) is the handler function when used as decorator,
            but Pathway passes it positionally, causing 'name' to be interpreted incorrectly.
            
            We need to:
            1. Detect if first arg is a callable (handler function)
            2. Extract name from kwargs (Pathway passes it as keyword)
            3. Filter unsupported kwargs
            4. Call original correctly
            """
            # FastMCP.tool() can be called two ways:
            # 1. As decorator: @mcp.tool(name="...") def handler(): ...
            # 2. Directly: mcp.tool(handler, name="...")
            
            # Check if first arg (after self) is a callable (handler function)
            handler_func = None
            if len(args) > 0:
                first_arg = args[0]
                if callable(first_arg):
                    # First arg is the handler function (decorator usage)
                    handler_func = first_arg
                    args = args[1:]  # Remove handler from args
            
            # Now extract name, description, tags from remaining args/kwargs
            filtered = {}
            
            # Handle name - Pathway passes it as keyword 'name', not positional
            # If there's a positional arg left, it might be name (but Pathway uses keyword)
            if 'name' in kwargs:
                filtered['name'] = kwargs['name']
            elif len(args) > 0:
                # Positional arg might be name (decorator usage without keyword)
                filtered['name'] = args[0]
                args = args[1:]
            
            # Handle description - map from 'title' if 'description' not provided
            if 'description' in kwargs:
                filtered['description'] = kwargs['description']
            elif 'title' in kwargs and kwargs.get('title'):
                # Use 'title' as 'description' if description not provided
                filtered['description'] = kwargs['title']
            
            # Handle tags
            if 'tags' in kwargs:
                filtered['tags'] = kwargs['tags']
            
            # If we have a handler function, FastMCP.tool() expects it as the first arg
            # when used as decorator: tool(name="...")(handler)
            if handler_func is not None:
                # Call as decorator: tool(name="...")(handler)
                decorator = original_tool(self, **filtered)
                return decorator(handler_func)
            else:
                # Call without handler (returns decorator)
                return original_tool(self, **filtered)
        
        # Replace the method on the class
        FastMCP.tool = patched_tool
        
        # Also patch any existing instances (though Pathway creates them later)
        # This ensures the patch works even if FastMCP instances exist
        sys.stderr.write("✓ FastMCP.tool() patched to filter unsupported arguments\n")
except (ImportError, AttributeError) as e:
    sys.stderr.write(f"Warning: Could not patch FastMCP.tool: {e}\n")
    pass

# Importing from explainibility_agents.mcp registers all MCP tools
from explainibility_agents.mcp import SmartFolioMCPServer

if __name__ == "__main__":
    server = SmartFolioMCPServer(app_id="smartfolio-xai", transport="sse", port=9123)
    
    print("Server configured for SSE on port 9123")
    
    try:
        server.serve()
    except KeyboardInterrupt:
        sys.stderr.write("Server stopped by user.\n")
    except Exception as e:
        sys.stderr.write(f"Server error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)