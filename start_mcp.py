import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
        sys.exit(1)