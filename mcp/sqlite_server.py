import sqlite3
import json
import logging
import os

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts.base import UserMessage, Message

DB_PATH = os.getenv("DB_PATH")

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sqlite_mcp_server")

DB_PATH = DB_PATH

# try connecting to the DB_PATH and log the outcome
try:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    logger.info(f"Connected to SQLite database at '{DB_PATH}'.")
except Exception as e:
    logger.exception(f"Failed to connect to SQLite database: {e}")
    raise

# initialise the MCP server
mcp = FastMCP("SQLite MCP Server")
logger.info("Initialized MCP server instance: SQLite MCP Server")

































if __name__ == "__main__":
    logger.info("Starting MCP server...")
    try:
        mcp.run()
    except Exception as e:
        logger.exception(f"MCP server terminated unexpectedly: {e}")
    finally:
        # close db connection on shutdown
        conn.close()
        logger.info("Database connection closed. MCP server shutdown complete.")