from __future__ import annotations

import csv
import dataclasses
import datetime as dt
from pathlib import Path

PRICE_COLUMNS = {
    "itinerary_id",
    "booking_date",
    "departure_date",
    "price",
}
ITINERARY_COLUMNS = {
    "itinerary_id",
    "origin",
    "destination",
    "flight_id",
    "frequency",
}
CO_BOOKING_COLUMNS = {"snapshot_date", "airport_a", "airport_b", "count"}


@dataclasses.dataclass(frozen=True)
class PriceRecord:
    itinerary_id: str
    booking_date: dt.date
    departure_date: dt.date
    price: float
    is_holiday: bool

    @property
    def booking_interval(self) -> int:
        return (self.departure_date - self.booking_date).days


@dataclasses.dataclass(frozen=True)
class ItineraryRecord:
    itinerary_id: str
    origin: str
    destination: str
    flight_id: str
    frequency: float


@dataclasses.dataclass(frozen=True)
class CoBookingRecord:
    snapshot_date: dt.date
    airport_a: str
    airport_b: str
    count: float


def _read_rows(path: str | Path, required: set[str]) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Required input file does not exist: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = sorted(required - fields)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {', '.join(missing)}")
        return list(reader)


def read_prices(path: str | Path) -> list[PriceRecord]:
    records = []
    for row in _read_rows(path, PRICE_COLUMNS):
        record = PriceRecord(
            itinerary_id=row["itinerary_id"],
            booking_date=dt.date.fromisoformat(row["booking_date"]),
            departure_date=dt.date.fromisoformat(row["departure_date"]),
            price=float(row["price"]),
            is_holiday=row.get("is_holiday", "0").strip().lower() in {"1", "true", "yes"},
        )
        if record.price <= 0:
            raise ValueError("Prices must be strictly positive")
        if record.booking_interval < 1:
            raise ValueError("departure_date must be later than booking_date")
        records.append(record)
    return records


def read_itineraries(path: str | Path) -> list[ItineraryRecord]:
    records = []
    seen = set()
    for row in _read_rows(path, ITINERARY_COLUMNS):
        record = ItineraryRecord(
            itinerary_id=row["itinerary_id"],
            origin=row["origin"],
            destination=row["destination"],
            flight_id=row["flight_id"],
            frequency=float(row["frequency"]),
        )
        if record.itinerary_id in seen:
            raise ValueError(f"Duplicate itinerary_id: {record.itinerary_id}")
        if record.frequency < 0:
            raise ValueError("frequency must be non-negative")
        seen.add(record.itinerary_id)
        records.append(record)
    return records


def read_co_bookings(path: str | Path) -> list[CoBookingRecord]:
    records = []
    for row in _read_rows(path, CO_BOOKING_COLUMNS):
        records.append(
            CoBookingRecord(
                snapshot_date=dt.date.fromisoformat(row["snapshot_date"]),
                airport_a=row["airport_a"],
                airport_b=row["airport_b"],
                count=float(row["count"]),
            )
        )
    return records
