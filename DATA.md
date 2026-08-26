# Data format

DCST-Net reads three CSV tables. Dates use ISO `YYYY-MM-DD` format, prices
must be positive and expressed in a consistent currency, and identifiers may
be arbitrary strings. No passenger or user identifier is needed for training.

## Price observations

`prices.csv` contains the observed itinerary prices.

| Column | Type | Description |
|---|---:|---|
| `itinerary_id` | string | Origin-destination-flight triplet identifier |
| `booking_date` | date | Date on which the price was observed |
| `departure_date` | date | Scheduled departure date |
| `price` | float | Observed ticket price |
| `is_holiday` | 0/1 | Optional slice label |

The booking interval is computed as `departure_date - booking_date`. Rows with
non-positive intervals or prices are rejected. At a request cutoff `b`, the
surface builder uses only observations whose booking dates are no later than
`b`.

## Itineraries

`itineraries.csv` defines the route and flight attached to each itinerary.

| Column | Type | Description |
|---|---:|---|
| `itinerary_id` | string | Key shared with `prices.csv` |
| `origin` | string | Origin airport identifier |
| `destination` | string | Destination airport identifier |
| `flight_id` | string | Flight identifier |
| `frequency` | float | Initial frequency value for standalone graphs |

For dated runs, graph frequency features are recomputed from price events
available at the graph refresh cutoff. The `frequency` column is used when a
standalone graph is built without timestamped price events.

## Co-booking snapshots

`co_bookings.csv` stores airport co-booking statistics accumulated by weekly
refresh jobs.

| Column | Type | Description |
|---|---:|---|
| `snapshot_date` | date | Completion cutoff of the weekly snapshot |
| `airport_a` | string | First airport |
| `airport_b` | string | Second airport |
| `count` | float | Co-booking count available at the cutoff |

For each request date, the graph loader selects the latest completed snapshot
that is not later than the request. Airport edges retain counts at or above the
configured percentile. Flight candidates are filtered by Jaccard similarity,
ranked by cosine similarity between degree-normalized frequency vectors, and
pruned to directed top-k neighborhoods with identifier-based tie breaking.
Flight-airport edges connect each flight to its origin and destination.

## Preparing another dataset

Before starting a run, check that the three files use the same itinerary keys,
that `itineraries.csv` has one row per itinerary, and that a completed graph
snapshot exists for every evaluation date. Split dates, currencies, and price
units should remain consistent within a configuration.

Input paths and split dates are set in the TOML configuration. The pipeline
records hashes of the inputs it reads but leaves the source CSV files in place.
Run outputs are written under the configured output directory.
