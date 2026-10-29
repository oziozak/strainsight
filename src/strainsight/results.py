"""Append-only run records, committed atomically in a local SQLite store."""
from contextlib import closing
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import sqlite3
from typing import Any


@dataclass(frozen=True)
class StrainSample:
    """Saved Green–Lagrange components at one reference-frame evaluation position."""

    frame: int
    x: float
    y: float
    exx: float | None
    eyy: float | None
    exy: float | None


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    configuration: dict[str, Any]
    metrics: dict[str, float | None]
    query_count: int
    runtime_seconds: float
    tracker_runtime_seconds: float | None = None
    peak_vram_bytes: int = 0
    strain_samples: dict[str, list[StrainSample]] = field(default_factory=dict)


def _decode_record(payload: str) -> RunRecord:
    data = json.loads(payload)
    data["strain_samples"] = {
        estimator: [StrainSample(**sample) for sample in samples]
        for estimator, samples in data.get("strain_samples", {}).items()}
    return RunRecord(**data)


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
        return ResultsReader(self.path.parent).read(run_id)


class ResultsReader:
    """Open an existing store using SQLite's read-only mode; never create one."""

    def __init__(self, directory: str | Path) -> None:
        self.path = Path(directory) / "runs.sqlite3"

    def read(self, run_id: str) -> RunRecord:
        with closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            row = connection.execute("SELECT record FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return _decode_record(row[0])

    def list_runs(self) -> list[RunRecord]:
        """Run summaries without strain samples; use read() to load a saved field."""
        with closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            rows = connection.execute(
                "SELECT json_remove(record, '$.strain_samples') FROM runs ORDER BY run_id").fetchall()
        return [_decode_record(row[0]) for row in rows]
