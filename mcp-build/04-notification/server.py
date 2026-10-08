import asyncio
from fastmcp import FastMCP,Context

mcp_client = FastMCP("notificationserver")

@mcp_client.tool
async def do_the_work(ctx:Context)->str:
    """Do a job in a slow way and report the progress along the way"""

    for step in range(1, 4):
        await asyncio.sleep(1)
        #notification 1 : a log message
        await ctx.info(f"Finished step {step}")

        # Notification 2: a progress update (progress out of total),
        # with a short text message that goes along with it.
        await ctx.report_progress(progress=step, total=3,
        message=f"Step {step} is done")



    # The normal answer. It arrives LAST, after all the notifications.
    return "Job done!"

if __name__ == "__main__":
    mcp_client.run(transport="stdio", show_banner=False)