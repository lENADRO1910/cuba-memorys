"""Database module: asyncpg connection pool and schema initialization.

Uses DATABASE_URL environment variable for connection.
Auto-creates the database AND tables on first run.
Zero manual setup required — fully autonomous.
"""

import asyncio
import json
import os
import sys
from importlib import resources
from typing import Any
from urllib.parse import urlparse, urlunparse

import asyncpg

# Connection pool singleton
_pool: asyncpg.Pool | None = None
_pool_lock = asyncio.Lock()
_semaphore = asyncio.Semaphore(3)  # Max 3 concurrent DB operations


def _get_admin_url_and_db() -> tuple[str, str]:
    """Parse DATABASE_URL into admin URL (pointing to 'postgres' db) and target db name.

    Returns:
        Tuple of (admin_url pointing to 'postgres' db, target database name).

    Raises:
        RuntimeError: If DATABASE_URL is not set.
    """
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError(
            "DATABASE_URL environment variable is required. "
            "Example: postgresql://user:pass@localhost:5432/brain"
        )

    parsed = urlparse(url)
    target_db = parsed.path.lstrip("/") or "brain"

    # Build admin URL pointing to default 'postgres' database
    admin_parsed = parsed._replace(path="/postgres")
    admin_url = urlunparse(admin_parsed)

    return admin_url, target_db


async def _ensure_database_exists() -> None:
    """Create the target database if it doesn't exist.

    Connects to the default 'postgres' database via direct PG connection
    (not PgBouncer) to run CREATE DATABASE. Idempotent and race-safe.
    """
    admin_url, target_db = _get_admin_url_and_db()

    try:
        conn = await asyncpg.connect(admin_url, timeout=10)
    except (OSError, asyncpg.PostgresError) as e:
        # Can't connect to admin DB — PostgreSQL might not be ready
        print(f"[cuba-memorys] Cannot connect to PostgreSQL: {e}", file=sys.stderr)
        return

    try:
        # Check if database exists
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1",
            target_db,
        )
        if not exists:
            # CREATE DATABASE cannot run inside a transaction
            await conn.execute(f'CREATE DATABASE "{target_db}"')
            print(f"[cuba-memorys] Created database '{target_db}'", file=sys.stderr)
        else:
            print(f"[cuba-memorys] Database '{target_db}' OK", file=sys.stderr)
    except asyncpg.DuplicateDatabaseError:
        # Race condition: another process created it between check and create
        pass
    finally:
        await conn.close()


async def get_pool() -> asyncpg.Pool:
    """Get or create the connection pool (lazy singleton).

    Auto-creates the database if it doesn't exist.

    Returns:
        asyncpg.Pool: Connection pool.

    Raises:
        RuntimeError: If DATABASE_URL is not set.
    """
    global _pool
    if _pool is not None:
        return _pool

    async with _pool_lock:
        if _pool is not None:
            return _pool

        # Step 1: Ensure the database exists
        await _ensure_database_exists()

        # Step 2: Connect to the target database
        database_url = os.environ.get("DATABASE_URL", "")
        if not database_url:
            raise RuntimeError(
                "DATABASE_URL environment variable is required. "
                "Example: postgresql://user:pass@localhost:5432/brain"
            )

        _pool = await asyncpg.create_pool(
            database_url,
            min_size=1,
            max_size=5,
            command_timeout=30,
        )
        return _pool


async def init_schema() -> None:
    """Initialize database schema (idempotent via IF NOT EXISTS).

    Creates extensions, tables, indexes, and triggers.
    Safe to call multiple times — all DDL uses IF NOT EXISTS.
    """
    pool = await get_pool()
    schema_sql = resources.files("cuba_memorys").joinpath("schema.sql").read_text()
    async with pool.acquire() as conn:
        await conn.execute(schema_sql)
    print("[cuba-memorys] Schema initialized", file=sys.stderr)


async def execute(query: str, *args: Any) -> str:
    """Execute a write query and return status.

    Args:
        query: SQL query with $1, $2, ... placeholders.
        *args: Query parameters.

    Returns:
        Status string from the query execution.
    """
    async with _semaphore:
        pool = await get_pool()
        async with pool.acquire() as conn:
            return await conn.execute(query, *args)


async def fetchrow(query: str, *args: Any) -> dict[str, Any] | None:
    """Fetch a single row as dict.

    Args:
        query: SQL query.
        *args: Query parameters.

    Returns:
        Dict with column names as keys, or None if no row found.
    """
    async with _semaphore:
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None


async def fetch(query: str, *args: Any) -> list[dict[str, Any]]:
    """Fetch multiple rows as list of dicts.

    Args:
        query: SQL query.
        *args: Query parameters.

    Returns:
        List of dicts with column names as keys.
    """
    async with _semaphore:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(r) for r in rows]


async def fetchval(query: str, *args: Any) -> Any:
    """Fetch a single value.

    Args:
        query: SQL query.
        *args: Query parameters.

    Returns:
        The first column of the first row.
    """
    async with _semaphore:
        pool = await get_pool()
        async with pool.acquire() as conn:
            return await conn.fetchval(query, *args)


async def close() -> None:
    """Close the connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def serialize(obj: Any) -> str:
    """JSON serialize with UUID/datetime support.

    Args:
        obj: Object to serialize.

    Returns:
        JSON string.
    """
    import datetime
    import uuid

    def _default(o: Any) -> Any:
        if isinstance(o, uuid.UUID):
            return str(o)
        if isinstance(o, (datetime.datetime, datetime.date)):
            return o.isoformat()
        raise TypeError(f"Not serializable: {type(o)}")

    return json.dumps(obj, default=_default)
