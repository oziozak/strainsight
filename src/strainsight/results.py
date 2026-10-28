"""Append-only run records, committed atomically in a local SQLite store."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sqlite3
from typing import Any


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    configuration: dict[str, Any]
    metrics: dict[str, float | None]
    query_count: int
    runtime_seconds: float
    tracker_runtime_seconds: float | None = None
    peak_vram_bytes: int = 0


class ResultsStore:
    def __init__(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "runs.sqlite3"
        with sqlite3.connect(self.path) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, record TEXT NOT NULL)")

    def write(self, record: RunRecord) -> None:
        payload = json.dumps(asdict(record), sort_keys=True, allow_nan=False)
        try:
            with sqlite3.connect(self.path) as connection:
                connection.execute("INSERT INTO runs VALUES (?, ?)", (record.run_id, payload))
        except sqlite3.IntegrityError as error:
            raise FileExistsError(f"immutable run already exists: {record.run_id}") from error

    def read(self, run_id: str) -> RunRecord:
        with sqlite3.connect(self.path) as connection:
            row = connection.execute("SELECT record FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return RunRecord(**json.loads(row[0]))
