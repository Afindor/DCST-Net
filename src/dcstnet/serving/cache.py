from __future__ import annotations

import contextlib
import json
import sqlite3
from collections.abc import Iterator, Sequence
from pathlib import Path


class TrajectoryCache:
    """SQLite-backed local implementation of the completed-refresh cache contract."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS trajectories (
                    itinerary_id TEXT NOT NULL,
                    departure_date TEXT NOT NULL,
                    booking_interval INTEGER NOT NULL,
                    refresh_version TEXT NOT NULL,
                    prices_json TEXT NOT NULL,
                    PRIMARY KEY (itinerary_id, departure_date, booking_interval, refresh_version)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS active_refresh (
                    namespace TEXT PRIMARY KEY,
                    refresh_version TEXT NOT NULL
                )
                """
            )

    @contextlib.contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def put(
        self,
        itinerary_id: str,
        departure_date: str,
        booking_interval: int,
        refresh_version: str,
        trajectory: Sequence[float],
    ) -> None:
        if not trajectory:
            raise ValueError("Cannot cache an empty trajectory")
        payload = json.dumps([float(value) for value in trajectory], separators=(",", ":"))
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO trajectories
                (itinerary_id, departure_date, booking_interval, refresh_version, prices_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (itinerary_id, departure_date, booking_interval, refresh_version, payload),
            )

    def get(
        self,
        itinerary_id: str,
        departure_date: str,
        booking_interval: int,
        refresh_version: str,
    ) -> list[float] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT prices_json FROM trajectories
                WHERE itinerary_id = ? AND departure_date = ?
                  AND booking_interval = ? AND refresh_version = ?
                """,
                (itinerary_id, departure_date, booking_interval, refresh_version),
            ).fetchone()
        return None if row is None else [float(value) for value in json.loads(row[0])]

    def publish_refresh(
        self,
        refresh_version: str,
        rows: Sequence[tuple[str, str, int, Sequence[float]]],
        namespace: str = "default",
    ) -> None:
        if not rows:
            raise ValueError("A refresh must contain at least one trajectory")
        payloads = []
        for itinerary_id, departure_date, booking_interval, trajectory in rows:
            if not trajectory:
                raise ValueError("Cannot publish an empty trajectory")
            payloads.append(
                (
                    itinerary_id,
                    departure_date,
                    booking_interval,
                    refresh_version,
                    json.dumps([float(value) for value in trajectory], separators=(",", ":")),
                )
            )
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO trajectories
                (itinerary_id, departure_date, booking_interval, refresh_version, prices_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                payloads,
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO active_refresh (namespace, refresh_version)
                VALUES (?, ?)
                """,
                (namespace, refresh_version),
            )

    def get_active(
        self,
        itinerary_id: str,
        departure_date: str,
        booking_interval: int,
        namespace: str = "default",
    ) -> list[float] | None:
        with self.connect() as connection:
            active = connection.execute(
                "SELECT refresh_version FROM active_refresh WHERE namespace = ?",
                (namespace,),
            ).fetchone()
        if active is None:
            return None
        return self.get(itinerary_id, departure_date, booking_interval, str(active[0]))
