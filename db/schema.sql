import sqlite3

con = sqlite3.connect("agent_project.db")
cur = con.cursor()

message_h_creation_query = """
    CREATE TABLE IF NOT EXISTS message_history (
        id INTEGER PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES sessions(session_id),
        role TEXT NOT NULL,
        content TEXT,
        timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
"""

token_usage_creation_query = """
    CREATE TABLE IF NOT EXISTS TOKEN_USAGE (
        id INTEGER PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES sessions(session_id),
        agent_name TEXT NOT NULL,
        prompt_tokens INTEGER NOT NULL,
        completion_tokens INTEGER NOT NULL,
        cost_euro FLOAT NOT NULL,
        timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
"""

sessions_creation_query = """
    CREATE TABLE IF NOT EXISTS SESSIONS (
        session_id TEXT PRIMARY KEY,
        timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
"""
cur.execute(message_h_creation_query)
cur.execute(token_usage_creation_query)
cur.execute(sessions_creation_query)

con.commit()
con.close()