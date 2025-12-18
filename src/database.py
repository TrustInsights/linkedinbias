# start src/database.py
"""Database operations for audit result persistence.

Provides SQLAlchemy Core schema definitions and functions for
storing and retrieving bias audit results in SQLite.
"""

import logging
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    insert,
    inspect,
    text,
)
from sqlalchemy.engine import Connection, Engine

from src.config import get_settings

metadata = MetaData()

audit_results = Table(
    "audit_results",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("male_name", String, nullable=False),
    Column("female_name", String, nullable=False),
    Column("cosine_similarity", Float, nullable=False),
    Column("bias_verdict", String, nullable=False),
    Column("timestamp", DateTime, default=datetime.utcnow, nullable=False),
    # Statistical columns (nullable for backward compatibility)
    Column("deviation_from_perfect", Float, nullable=True),
    Column("z_score", Float, nullable=True),
    Column("percentile", Float, nullable=True),
)


def get_db_engine() -> Engine:
    """Create and return a SQLAlchemy engine for the audit database.

    Creates the output directory if it doesn't exist and returns
    an engine configured for the SQLite database.

    Returns:
        SQLAlchemy Engine connected to the audit database.
    """
    settings = get_settings()
    # Create output directory if it doesn't exist
    settings.paths.output_dir.mkdir(exist_ok=True)

    db_path = settings.paths.output_dir / "audit.db"
    return create_engine(f"sqlite:///{db_path}")


def _migrate_audit_results_table(engine: Engine) -> None:
    """Add missing columns to existing audit_results table.

    Handles backward compatibility by adding new statistical columns
    to databases created before these columns were added to the schema.

    Args:
        engine: SQLAlchemy engine connected to the database.
    """
    inspector = inspect(engine)

    # Check if table exists
    if "audit_results" not in inspector.get_table_names():
        return  # Table doesn't exist yet, will be created by create_all

    # Get existing columns
    existing_columns = {col["name"] for col in inspector.get_columns("audit_results")}

    # Define columns that may need to be added (name -> SQL type)
    new_columns = {
        "deviation_from_perfect": "FLOAT",
        "z_score": "FLOAT",
        "percentile": "FLOAT",
    }

    # Add missing columns
    with engine.connect() as conn:
        for col_name, col_type in new_columns.items():
            if col_name not in existing_columns:
                logging.info(f"Migrating database: adding column '{col_name}'")
                conn.execute(
                    text(f"ALTER TABLE audit_results ADD COLUMN {col_name} {col_type}")
                )
        conn.commit()


def init_db(engine: Engine) -> None:
    """Initialize the database schema.

    Creates all tables defined in the metadata if they don't exist.
    Also migrates existing tables to add any new columns.

    Args:
        engine: SQLAlchemy engine to use for table creation.
    """
    # First, migrate existing tables to add new columns if needed
    _migrate_audit_results_table(engine)

    # Then create any missing tables
    metadata.create_all(engine)


def save_result(conn: Connection, data: dict[str, Any]) -> None:
    """Save a single audit result to the database.

    Args:
        conn: Active database connection.
        data: Dictionary containing audit result fields matching
            the audit_results table schema.
    """
    stmt = insert(audit_results).values(**data)
    conn.execute(stmt)


def clear_results(engine: Engine) -> int:
    """Clear all existing audit results from the database.

    Use this before starting a new audit run to avoid duplicate records
    and ensure clean statistical analysis.

    Args:
        engine: SQLAlchemy engine connected to the database.

    Returns:
        Number of records deleted.
    """
    with engine.connect() as conn:
        result = conn.execute(text("DELETE FROM audit_results"))
        conn.commit()
        return result.rowcount


def get_result_count(engine: Engine) -> int:
    """Get the count of existing audit results.

    Args:
        engine: SQLAlchemy engine connected to the database.

    Returns:
        Number of records in audit_results table.
    """
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM audit_results"))
        row = result.fetchone()
        return row[0] if row else 0


# end src/database.py
