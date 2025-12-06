import asyncio
import os
from pathlib import Path

from fastmcp import Client
import os
os.environ['OPENAI_API_KEY'] = "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"

BASE_DIR = Path(__file__).resolve().parent


def default_path(env_var: str, relative_path: str) -> str:
    candidate = os.environ.get(env_var)
    if candidate:
        return candidate
    return str((BASE_DIR / relative_path).resolve())


DEFAULT_ARGS = {
    "date": "2024-12-26",
    "monthly_log_csv": default_path(
        "SMARTFOLIO_MONTHLY_LOG",
        "logs/monthly/2024-12/final_test_weights_20251204_105206.csv",
    ),
    "model_path": default_path(
        "SMARTFOLIO_MODEL_PATH",
        "checkpoints_risk05/ppo_hgat_custom_20251204_105206.zip",
    ),
    "market": "custom",
    "data_root": default_path("SMARTFOLIO_DATA_ROOT", "dataset_default"),
    "top_k": 5,
    "lookback_days": 30,
    "monthly_run_id": None,
    "output_dir": default_path("SMARTFOLIO_OUTPUT_DIR", "explainability_results/latest_run"),
    "llm": False,
    "llm_model": "gpt-5-mini",
    "latent": True,
}

SERVER_URL = os.environ.get("SMARTFOLIO_MCP_URL", "http://localhost:9123/mcp/")

def get_user_input(current_args):
    """Prompts the user to override default arguments."""
    print("\n--- Configure SmartFolio XAI Run ---")
    print("Press Enter to keep [current value]. Type 'run' to execute immediately. Type 'exit' to quit.\n")
    
    args = current_args.copy()
    
    action = input(f"Ready to run with date={args['date']}? (Press Enter to run, 'c' to configure, 'q' to quit): ").strip().lower()
    if action == 'q':
        return None
    if action != 'c':
        return args

    for key, default_val in args.items():
        prompt = f"{key} [{default_val}]: "
        user_val = input(prompt).strip()
        
        if user_val:
            if user_val.lower() == 'exit': return None
            
            if isinstance(default_val, bool):
                if user_val.lower() in ('y', 'yes', 'true', 't', '1'):
                    args[key] = True
                elif user_val.lower() in ('n', 'no', 'false', 'f', '0'):
                    args[key] = False
            elif isinstance(default_val, int):
                try:
                    args[key] = int(user_val)
                except ValueError:
                    print(f"Invalid integer for {key}, keeping default.")
            elif default_val is None:
                if user_val.lower() == "none":
                    args[key] = None
                else:
                    args[key] = user_val
            else:
                args[key] = user_val
    
    return args

async def run():
    print(" Connecting to SmartFolio MCP Server (Streamable HTTP)...")
    print("   Ensure 'python3 start_mcp.py' is running in another terminal!")
    
    client = Client(SERVER_URL)

    async with client:
        print(" Server Connected!")
        
        current_args = DEFAULT_ARGS.copy()
        
        while True:
            run_args = get_user_input(current_args)
            if run_args is None:
                print("Exiting...")
                break
            
            current_args = run_args
            
            print(f"\n  Executing 'run_xai_orchestrator' for {run_args['date']}...")
            
            try:
                result = await client.call_tool("run_xai_orchestrator", arguments=run_args)
                
                print("\n Tool Execution Successful!")
                print("-" * 40)
                if hasattr(result, 'content'):
                    for content in result.content:
                        if hasattr(content, 'text'):
                            print(content.text)
                        else:
                            print(content)
                else:
                    print(result)
                print("-" * 40)
                    
            except Exception as e:
                print(f"\n Tool Execution Failed: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")