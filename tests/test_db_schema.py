import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# db/schema.sql (W3-01)
# ---------------------------------------------------------------------------

SCHEMA_SCRIPT = Path(__file__).resolve().parent.parent / "db" / "schema.sql"


def run_schema_script(cwd: Path) -> subprocess.CompletedProcess:
    """Run the schema script in `cwd`, where it creates agent_project.db."""
    return subprocess.run(
        [sys.executable, str(SCHEMA_SCRIPT)],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def db(tmp_path):
    """A connection to a fresh database created by the schema script.

    Foreign keys are switched on because SQLite leaves them off by default,
    and the FK tests below rely on them being enforced.
    """
    result = run_schema_script(tmp_path)
    assert result.returncode == 0, result.stderr
    con = sqlite3.connect(tmp_path / "agent_project.db")
    con.execute("PRAGMA foreign_keys = ON")
    yield con
    con.close()


def insert_session(con, session_id="s1"):
    con.execute("INSERT INTO sessions (session_id) VALUES (?)", (session_id,))


def test_creates_expected_tables(db):
    rows = db.execute(
        "SELECT lower(name) FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    tables = {name for (name,) in rows}
    assert {"sessions", "message_history", "token_usage"} <= tables


def test_script_is_idempotent(tmp_path):
    # IF NOT EXISTS means a second run over the same db must not fail.
    first = run_schema_script(tmp_path)
    assert first.returncode == 0, first.stderr
    second = run_schema_script(tmp_path)
    assert second.returncode == 0, second.stderr


def test_session_id_is_primary_key(db):
    insert_session(db)
    with pytest.raises(sqlite3.IntegrityError):
        insert_session(db)


def test_multiple_messages_per_session(db):
    insert_session(db)
    for role, content in [("user", "hi"), ("assistant", "hello"), ("user", "bye")]:
        db.execute(
            "INSERT INTO message_history (session_id, role, content) VALUES (?, ?, ?)",
            ("s1", role, content),
        )
    (count,) = db.execute(
        "SELECT count(*) FROM message_history WHERE session_id = 's1'"
    ).fetchone()
    assert count == 3


def test_message_history_rejects_unknown_session(db):
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO message_history (session_id, role) VALUES (?, ?)",
            ("no-such-session", "user"),
        )


def test_token_usage_roundtrip(db):
    insert_session(db)
    db.execute(
        """INSERT INTO token_usage
           (session_id, agent_name, prompt_tokens, completion_tokens, cost_euro)
           VALUES (?, ?, ?, ?, ?)""",
        ("s1", "researcher", 120, 45, 0.0031),
    )
    row = db.execute(
        """SELECT agent_name, prompt_tokens, completion_tokens, cost_euro
           FROM token_usage WHERE session_id = 's1'"""
    ).fetchone()
    assert row == ("researcher", 120, 45, 0.0031)


def test_token_usage_rejects_unknown_session(db):
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            """INSERT INTO token_usage
               (session_id, agent_name, prompt_tokens, completion_tokens, cost_euro)
               VALUES (?, ?, ?, ?, ?)""",
            ("no-such-session", "researcher", 1, 1, 0.0),
        )


def test_timestamps_default_to_current_time(db):
    insert_session(db)
    db.execute(
        "INSERT INTO message_history (session_id, role) VALUES ('s1', 'user')"
    )
    for table in ("sessions", "message_history"):
        (ts,) = db.execute(f"SELECT timestamp FROM {table}").fetchone()
        assert ts is not None and ts != ""


def test_role_is_required(db):
    insert_session(db)
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO message_history (session_id, content) VALUES ('s1', 'hi')"
        )
