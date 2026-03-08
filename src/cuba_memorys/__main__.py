"""Entry point: python -m cuba_memorys or cuba-memorys CLI."""

import asyncio
import sys

from cuba_memorys.server import main as server_main


def main() -> None:
    """Run the Cuba-Memorys MCP server."""
    try:
        asyncio.run(server_main())
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
