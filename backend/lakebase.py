"""Lakebase (managed Postgres) connection pool for the backend.

Lakebase issues short-lived OAuth tokens (~1h). The pool's ``max_lifetime``
recycles connections before tokens expire so the agent never sees an
auth-related disconnect mid-turn. Conninfo is materialised lazily via
``db_client.build_lakebase_conninfo`` — calling it again refreshes the token
on the next pool-fill.

The pool is used by ``session_manager`` (P8) to persist session metadata so
backend restarts don't drop the session list, and the dashboard can join
sessions to traces in MLflow.

If ``ML_INTERN_LAKEBASE_INSTANCE`` isn't configured (local dev, unit tests),
``init`` is a no-op and ``get_pool()`` returns None — callers should check.
"""

from __future__ import annotations

import logging
from typing import Optional

from agent.config import Config
from agent.core import db_client

logger = logging.getLogger(__name__)

_POOL = None  # type: ignore[var-annotated]


def init(config: Config) -> bool:
    """Build the connection pool. Idempotent. Returns True on success."""
    global _POOL
    if _POOL is not None:
        return True
    settings = db_client.resolve_settings(config)
    if not settings.lakebase_instance:
        logger.info("Lakebase not configured — skipping pool init.")
        return False
    try:
        from psycopg_pool import ConnectionPool

        def _conninfo() -> str:
            # Re-resolved on each refill so the OAuth token can rotate.
            return db_client.build_lakebase_conninfo(settings)

        _POOL = ConnectionPool(
            conninfo=_conninfo(),
            min_size=1,
            max_size=10,
            max_lifetime=2700,  # 45 min — tokens last ~1h, leave a margin
            kwargs={"autocommit": True},
            open=True,
        )
        _ensure_schema(_POOL)
        logger.info("Lakebase pool initialised (instance=%s).", settings.lakebase_instance)
        return True
    except Exception as e:
        logger.warning("Lakebase pool init failed (%s) — proceeding without persistence.", e)
        _POOL = None
        return False


def shutdown() -> None:
    global _POOL
    if _POOL is None:
        return
    try:
        _POOL.close()
    except Exception as e:
        logger.debug("Lakebase pool close suppressed: %s", e)
    _POOL = None


def get_pool():
    """Return the active pool or None if Lakebase isn't configured."""
    return _POOL


def _ensure_schema(pool) -> None:
    """Create the tables the backend persists to. Idempotent."""
    with pool.connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_intern_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                user_email TEXT,
                model_name TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                last_active_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                message_count INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS ml_intern_sessions_user_idx
                ON ml_intern_sessions(user_id, last_active_at DESC)
        """)


def upsert_session(*, session_id: str, user_id: str, user_email: str | None,
                   model_name: str, message_count: int = 0,
                   is_active: bool = True) -> None:
    pool = get_pool()
    if pool is None:
        return
    try:
        with pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO ml_intern_sessions
                  (session_id, user_id, user_email, model_name, message_count, is_active)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (session_id) DO UPDATE SET
                    user_email = EXCLUDED.user_email,
                    model_name = EXCLUDED.model_name,
                    message_count = EXCLUDED.message_count,
                    is_active = EXCLUDED.is_active,
                    last_active_at = now()
                """,
                (session_id, user_id, user_email, model_name, message_count, is_active),
            )
    except Exception as e:
        logger.debug("upsert_session suppressed: %s", e)


def mark_session_inactive(session_id: str) -> None:
    pool = get_pool()
    if pool is None:
        return
    try:
        with pool.connection() as conn:
            conn.execute(
                "UPDATE ml_intern_sessions SET is_active = FALSE, last_active_at = now() "
                "WHERE session_id = %s",
                (session_id,),
            )
    except Exception as e:
        logger.debug("mark_session_inactive suppressed: %s", e)
