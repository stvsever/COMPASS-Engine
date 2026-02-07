"""SQLite-backed embedding cache for reusable COMPASS retrieval logic."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import tempfile
import logging
from pathlib import Path
from typing import Callable, List, Optional


def _default_db_path() -> Path:
    override = os.getenv("COMPASS_EMBEDDING_DB_PATH")
    if override:
        return Path(override).expanduser()

    home = Path.home()
    if os.name == "nt":
        local = os.getenv("LOCALAPPDATA")
        base = Path(local) if local else (home / "AppData" / "Local")
        return base / "compass" / "embeddings.sqlite3"
    if os.uname().sysname.lower() == "darwin":
        return home / "Library" / "Caches" / "compass" / "embeddings.sqlite3"
    return home / ".cache" / "compass" / "embeddings.sqlite3"


class EmbeddingStore:
    """Thread-safe SQLite embedding store with global and participant scopes."""

    def __init__(self, db_path: Optional[Path] = None):
        self.fallback_reason: Optional[str] = None
        initial_path = (db_path or _default_db_path()).expanduser()
        self.db_path = initial_path
        db_path_str = str(initial_path)
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.fallback_reason = f"default_path_unwritable:{exc}"
            try:
                fallback_dir = Path(tempfile.gettempdir()) / "compass"
                fallback_dir.mkdir(parents=True, exist_ok=True)
                self.db_path = fallback_dir / "embeddings.sqlite3"
                db_path_str = str(self.db_path)
            except Exception as fallback_exc:
                self.fallback_reason += f";tmp_fallback_failed:{fallback_exc}"
                self.db_path = Path(":memory:")
                db_path_str = ":memory:"
        if self.fallback_reason:
            logger.warning("Embedding store fallback active: %s", self.fallback_reason)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path_str, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute("PRAGMA temp_store=MEMORY;")
            self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS global_path_embeddings (
                text_hash TEXT NOT NULL,
                text_value TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                source_type TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_used_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                hit_count INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (text_hash, embedding_model)
            );

            CREATE TABLE IF NOT EXISTS participant_embeddings (
                participant_id TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                text_value TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                source_type TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_used_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                hit_count INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (participant_id, text_hash, embedding_model)
            );

            CREATE INDEX IF NOT EXISTS idx_global_source
            ON global_path_embeddings (source_type);

            CREATE INDEX IF NOT EXISTS idx_participant_scope
            ON participant_embeddings (participant_id, source_type);
            """
        )
        self._conn.commit()

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

    def _lookup_global(self, text_hash: str, model: str) -> Optional[List[float]]:
        row = self._conn.execute(
            """
            SELECT embedding_json
            FROM global_path_embeddings
            WHERE text_hash=? AND embedding_model=?
            """,
            (text_hash, model),
        ).fetchone()
        if not row:
            return None
        self._conn.execute(
            """
            UPDATE global_path_embeddings
            SET hit_count = hit_count + 1,
                last_used_at = CURRENT_TIMESTAMP
            WHERE text_hash=? AND embedding_model=?
            """,
            (text_hash, model),
        )
        self._conn.commit()
        return json.loads(row["embedding_json"])

    def _lookup_participant(self, participant_id: str, text_hash: str, model: str) -> Optional[List[float]]:
        row = self._conn.execute(
            """
            SELECT embedding_json
            FROM participant_embeddings
            WHERE participant_id=? AND text_hash=? AND embedding_model=?
            """,
            (participant_id, text_hash, model),
        ).fetchone()
        if not row:
            return None
        self._conn.execute(
            """
            UPDATE participant_embeddings
            SET hit_count = hit_count + 1,
                last_used_at = CURRENT_TIMESTAMP
            WHERE participant_id=? AND text_hash=? AND embedding_model=?
            """,
            (participant_id, text_hash, model),
        )
        self._conn.commit()
        return json.loads(row["embedding_json"])

    def _upsert_global(
        self,
        text_hash: str,
        text_value: str,
        embedding: List[float],
        model: str,
        source_type: str,
    ) -> None:
        payload = json.dumps(embedding)
        self._conn.execute(
            """
            INSERT INTO global_path_embeddings (
                text_hash, text_value, embedding_json, embedding_model, dimension, source_type
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(text_hash, embedding_model) DO UPDATE SET
                text_value=excluded.text_value,
                embedding_json=excluded.embedding_json,
                dimension=excluded.dimension,
                source_type=excluded.source_type,
                updated_at=CURRENT_TIMESTAMP,
                last_used_at=CURRENT_TIMESTAMP,
                hit_count=global_path_embeddings.hit_count + 1
            """,
            (text_hash, text_value, payload, model, len(embedding), source_type),
        )
        self._conn.commit()

    def _upsert_participant(
        self,
        participant_id: str,
        text_hash: str,
        text_value: str,
        embedding: List[float],
        model: str,
        source_type: str,
    ) -> None:
        payload = json.dumps(embedding)
        self._conn.execute(
            """
            INSERT INTO participant_embeddings (
                participant_id, text_hash, text_value, embedding_json, embedding_model, dimension, source_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(participant_id, text_hash, embedding_model) DO UPDATE SET
                text_value=excluded.text_value,
                embedding_json=excluded.embedding_json,
                dimension=excluded.dimension,
                source_type=excluded.source_type,
                updated_at=CURRENT_TIMESTAMP,
                last_used_at=CURRENT_TIMESTAMP,
                hit_count=participant_embeddings.hit_count + 1
            """,
            (participant_id, text_hash, text_value, payload, model, len(embedding), source_type),
        )
        self._conn.commit()

    def get_or_create_global(
        self,
        text: str,
        model: str,
        embed_fn: Callable[[str, str], List[float]],
        source_type: str = "feature_path",
    ) -> List[float]:
        text_hash = self._hash_text(text)
        with self._lock:
            cached = self._lookup_global(text_hash, model)
        if cached is not None:
            return cached

        embedding = embed_fn(text, model)
        with self._lock:
            self._upsert_global(text_hash, text, embedding, model, source_type)
        return embedding

    def get_or_create_participant(
        self,
        participant_id: str,
        text: str,
        model: str,
        embed_fn: Callable[[str, str], List[float]],
        source_type: str = "participant_context",
    ) -> List[float]:
        pid = str(participant_id or "unknown")
        text_hash = self._hash_text(text)
        with self._lock:
            cached = self._lookup_participant(pid, text_hash, model)
        if cached is not None:
            return cached

        embedding = embed_fn(text, model)
        with self._lock:
            self._upsert_participant(pid, text_hash, text, embedding, model, source_type)
        return embedding


_store_instance: Optional[EmbeddingStore] = None


def get_embedding_store() -> EmbeddingStore:
    global _store_instance
    if _store_instance is None:
        _store_instance = EmbeddingStore()
    return _store_instance
logger = logging.getLogger("compass.embedding_store")
