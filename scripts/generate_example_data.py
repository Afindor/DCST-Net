from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import math
from pathlib import Path

ITINERARIES = (
    ("ODF001", "BJS", "SHA", "CA100", 1800),
    ("ODF002", "BJS", "CAN", "CA200", 1350),
    ("ODF003", "SHA", "CAN", "MU300", 1250),
    ("ODF004", "SHA", "BJS", "MU400", 1600),
)


def _noise(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return (int.from_bytes(digest[:2], "big") / 65535.0 - 0.5) * 8.0


def generate(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "itineraries.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("itinerary_id", "origin", "destination", "flight_id", "frequency"))
        writer.writerows(ITINERARIES)

    airports = ("BJS", "SHA", "CAN")
    snapshots = [dt.date(2023, 12, 31) + dt.timedelta(days=7 * step) for step in range(8)]
    with (destination / "co_bookings.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("snapshot_date", "airport_a", "airport_b", "count"))
        for snapshot_index, snapshot in enumerate(snapshots):
            for left_index, left in enumerate(airports):
                for right in airports[left_index + 1 :]:
                    base = {("BJS", "SHA"): 240, ("BJS", "CAN"): 170, ("SHA", "CAN"): 150}[
                        (left, right)
                    ]
                    writer.writerow((snapshot.isoformat(), left, right, base + 4 * snapshot_index))

    departure_start = dt.date(2024, 1, 15)
    departure_end = dt.date(2024, 2, 18)
    base_price = {"ODF001": 810.0, "ODF002": 940.0, "ODF003": 760.0, "ODF004": 835.0}
    with (destination / "prices.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("itinerary_id", "booking_date", "departure_date", "price", "is_holiday"))
        departure = departure_start
        while departure <= departure_end:
            holiday = int(dt.date(2024, 2, 1) <= departure <= dt.date(2024, 2, 4))
            calendar_effect = 22.0 if departure.weekday() in {4, 5} else 0.0
            calendar_effect += 38.0 * holiday
            day_index = (departure - departure_start).days
            for itinerary_id, _, _, _, _ in ITINERARIES:
                for interval in range(1, 15):
                    booking_date = departure - dt.timedelta(days=interval)
                    booking_pressure = 2.8 * (14 - interval)
                    market_wave = 42.0 * math.sin((day_index + 1.7 * interval) / 3.6)
                    price = (
                        base_price[itinerary_id]
                        + calendar_effect
                        + booking_pressure
                        + market_wave
                        + _noise(f"{itinerary_id}|{departure}|{interval}")
                    )
                    writer.writerow(
                        (
                            itinerary_id,
                            booking_date.isoformat(),
                            departure.isoformat(),
                            f"{price:.2f}",
                            holiday,
                        )
                    )
            departure += dt.timedelta(days=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the deterministic public example corpus")
    parser.add_argument("--output", type=Path, default=Path("data/example"))
    arguments = parser.parse_args()
    generate(arguments.output)


if __name__ == "__main__":
    main()
