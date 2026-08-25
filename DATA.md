# Data contract

All dates use ISO `YYYY-MM-DD` format. Prices are positive values in one
consistent currency. Identifiers may be pseudonymous strings; the model never
requires passenger or user identity.

## `prices.csv`

| Column | Type | Meaning |
|---|---:|---|
| `itinerary_id` | string | Origin-destination-flight triplet identifier |
| `booking_date` | date | Date on which the price was observed |
| `departure_date` | date | Scheduled departure date |
| `price` | float | Observed ticket price |
| `is_holiday` | 0/1 | Optional analysis-slice indicator |

The booking interval is derived as `departure_date - booking_date`. A cell is
admissible for a request cutoff `b` only when its own booking date is no later
than `b`. The loader rejects non-positive intervals and prices.

## `itineraries.csv`

| Column | Type | Meaning |
|---|---:|---|
| `itinerary_id` | string | Key shared with `prices.csv` |
| `origin` | string | Origin airport identifier |
| `destination` | string | Destination airport identifier |
| `flight_id` | string | Flight identifier |
| `frequency` | float | Cold-start frequency prior |

During an experiment, graph frequency features are recomputed from price events
whose booking dates are no later than the selected graph refresh cutoff. The
`frequency` field is used only when an external caller builds a standalone
graph without timestamped price events.

## `co_bookings.csv`

| Column | Type | Meaning |
|---|---:|---|
| `snapshot_date` | date | Completion cutoff of the weekly snapshot |
| `airport_a` | string | First airport |
| `airport_b` | string | Second airport |
| `count` | float | Historical co-booking count available by the cutoff |

The graph store selects the latest completed `snapshot_date` no later than each
request date. Airport edges retain counts at or above the configured percentile.
Flight edges use Jaccard candidate filtering, cosine ranking of degree-normalized
frequency vectors, deterministic identifier tie-breaking, and directed top-k
pruning. Flight-airport links connect each flight to its origin and destination.

## Private-data preparation

Before a full run, verify that:

1. each file contains only information available by its timestamp;
2. the configured split dates match the intended corpus;
3. duplicate itinerary identifiers are removed from `itineraries.csv`;
4. currency and price units are consistent within a run;
5. every test request has a completed graph snapshot at or before its date.

The pipeline writes a configuration digest and SHA-256 hashes for all consumed
input and produced artifacts. It does not copy raw inputs into the output tree.
