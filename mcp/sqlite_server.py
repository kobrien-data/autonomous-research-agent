import json
import logging
import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# MCP clients spawn this server from an arbitrary cwd, so resolve .env
# relative to this file rather than the working directory
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DB_PATH = os.getenv("DB_PATH")
if not DB_PATH:
    raise RuntimeError("DB_PATH is not set — define it in .env or the MCP client config")

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sqlite_mcp_server")

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

# Tool lists all tables in the database
@mcp.tool()
def list_tables() -> str:
    """
    List all tables in the database with their CREATE statements.
    """
    rows = conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' and name NOT LIKE 'sqlite_%'"
    )
    return json.dumps([dict(r) for r in rows])

# Tool to get the schema for each table
@mcp.tool()
def describe_table(table_name: str) -> str:
    """Get column names, types, and constraints for a table."""
    # PRAGMA doesn't support parameter binding, so validate the name against
    # the actual table list instead of interpolating untrusted input
    tables = {
        r["name"]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    if table_name not in tables:
        raise ValueError(f"Unknown table: {table_name}")
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return json.dumps([dict(r) for r in rows])

# Tool to run arbitrary read-only queries
@mcp.tool()
def read_query(query: str, limit: int = 100) -> str:
    """Run a read-only SELECT query against the database. Results are capped at `limit` rows."""
    if not query.lstrip().upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed")
    rows = conn.execute(query).fetchmany(limit)
    return json.dumps([dict(r) for r in rows])

def _ensure_session(session_id: str) -> None:
    """Create the session row if it doesn't exist yet."""
    conn.execute(
        "INSERT OR IGNORE INTO sessions (session_id) VALUES (?)", (session_id,)
    )

# Tool to save a message to the conversation history
@mcp.tool()
def save_message(session_id: str, role: str, content: str) -> str:
    """Save a message to the conversation history for a session."""
    _ensure_session(session_id)
    cur = conn.execute(
        "INSERT INTO message_history (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, content),
    )
    conn.commit()
    return json.dumps({"id": cur.lastrowid, "session_id": session_id})

# Tool to retrieve the conversation history for a session
@mcp.tool()
def get_history(session_id: str, limit: int = 50) -> str:
    """Get the last `limit` messages for a session, in chronological order."""
    rows = conn.execute(
        """
        SELECT * FROM (
            SELECT id, role, content, timestamp
            FROM message_history
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
        ) ORDER BY id ASC
        """,
        (session_id, limit),
    ).fetchall()
    return json.dumps([dict(r) for r in rows])

# Tool to record token usage for an agent call
@mcp.tool()
def save_token_usage(
    session_id: str,
    agent_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost_euro: float,
) -> str:
    """Record token usage and cost for an agent call in a session."""
    _ensure_session(session_id)
    cur = conn.execute(
        """
        INSERT INTO token_usage (session_id, agent_name, prompt_tokens, completion_tokens, cost_euro)
        VALUES (?, ?, ?, ?, ?)
        """,
        (session_id, agent_name, prompt_tokens, completion_tokens, cost_euro),
    )
    conn.commit()
    return json.dumps({"id": cur.lastrowid, "session_id": session_id})

# Tool to get aggregate stats for a session
@mcp.tool()
def get_session_stats(session_id: str) -> str:
    """Get message count, token totals, and cost for a session, with a per-agent breakdown."""
    message_count = conn.execute(
        "SELECT COUNT(*) AS n FROM message_history WHERE session_id = ?",
        (session_id,),
    ).fetchone()["n"]
    totals = conn.execute(
        """
        SELECT
            COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
            COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
            COALESCE(SUM(cost_euro), 0) AS cost_euro
        FROM token_usage WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    per_agent = conn.execute(
        """
        SELECT
            agent_name,
            SUM(prompt_tokens) AS prompt_tokens,
            SUM(completion_tokens) AS completion_tokens,
            SUM(cost_euro) AS cost_euro
        FROM token_usage WHERE session_id = ?
        GROUP BY agent_name
        """,
        (session_id,),
    ).fetchall()
    return json.dumps(
        {
            "session_id": session_id,
            "message_count": message_count,
            "totals": dict(totals),
            "per_agent": [dict(r) for r in per_agent],
        }
    )

# Tool to delete all data for a session
@mcp.tool()
def clear_session(session_id: str) -> str:
    """Delete all messages, token usage, and the session row for a session."""
    messages_deleted = conn.execute(
        "DELETE FROM message_history WHERE session_id = ?", (session_id,)
    ).rowcount
    usage_deleted = conn.execute(
        "DELETE FROM token_usage WHERE session_id = ?", (session_id,)
    ).rowcount
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    return json.dumps(
        {
            "session_id": session_id,
            "messages_deleted": messages_deleted,
            "usage_rows_deleted": usage_deleted,
        }
    )



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