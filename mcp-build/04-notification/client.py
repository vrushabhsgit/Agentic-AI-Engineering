from langchain_mcp_adapters.callbacks import Callbacks
from langchain_mcp_adapters.client import MultiServerMCPClient
import sys
from pathlib import Path
SERVER_PATH = Path(__file__).parent/"server.py"

async def on_log(params, context):
    print(
        f"LOG: [{context.server_name} / {context.tool_name}]",
        params.data["msg"]
    )


async def on_progress(progress, total, message, context):
    print(
        f"PROGRESS: [{context.server_name} / {context.tool_name}] "
        f"{progress:.0f} / {total:.0f} - {message}"
    )


client = MultiServerMCPClient(
    {
        "cart_server": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(SERVER_PATH)]
        }
    },
    callbacks=Callbacks(
        on_logging_message=on_log,
        on_progress=on_progress
    )
)