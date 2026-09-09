# Storage and proposed retention

Aggregation, aggregate history APIs, read-only auditing and manual verified
raw-history cleanup are implemented. No cleanup schedule is enabled. The periods
below describe the storage policy; only raw cleanup has a deletion command.

## Local measurement

Snapshot: September 8, 2026, 12:38 UTC. These are local development figures,
not production measurements. Repeat with [storage SQL](../scripts/sql/observation-storage.sql).

| Table | Rows | Table allocation | Index allocation | Total |
| --- | ---: | ---: | ---: | ---: |
| Solar wind | 13,347 | 11,837,440 B | 1,245,184 B | 12.48 MiB |
| Kp/Dst | 234 | 73,728 B | 16,384 B | 88 KiB |
| Source status | 4 | 40,960 B | 16,384 B | 56 KiB |

Solar wind covers September 6–8, with one full UTC day. September 7 contains
7,305 rows across magnetic and plasma feeds and three spacecraft. All spacecraft
are stored; public history selects the active stream.

Magnetic rows average about 694 B and plasma rows 938–1,008 B. Raw JSON accounts
for 547 B and 814–885 B respectively. Total allocation including indexes averages
about 980 B per row in this snapshot.

## Estimated growth

| Solar wind scenario | Daily growth | 90 days | One year |
| --- | ---: | ---: | ---: |
| Observed 7,305 rows/day | 6.83 MiB | 0.60 GiB | 2.43 GiB |
| Three spacecraft × two feeds × 1,440 minutes | 8.08 MiB | 0.71 GiB | 2.88 GiB |

These extrapolations use a short, incomplete sample. They exclude aggregates,
WAL, backups, replicas and other tables. Measure production for 7–14 days before
setting a storage budget or enabling cleanup.

Kp/Dst add 32 native intervals/day, or 11,680/year. Their row payload is roughly
1.73 MiB/year before indexes and page overhead.

## Proposed periods

| Dataset | Retention |
| --- | --- |
| All spacecraft minute solar wind, including raw records | 90 days |
| Active solar wind five-minute aggregates | Two years |
| Active solar wind hourly aggregates | No expiry initially |
| Native Kp/Dst and raw records | No expiry initially |
| Collector status | Current state and last error |

Aggregates preserve the selected active series, not every spacecraft. Deleting
raw data removes the ability to reconstruct individual samples or choose another
spacecraft later.

## Before enabling deletion

1. Measure production growth and aggregate sizes.
2. Run the [audit](aggregation-audit.md) on candidate date ranges.
3. Require verified aggregates and completed processing for each range.
4. Recheck revisions and queued work when deleting; an earlier audit is not a
   lasting cleanup permission.

[Aggregation semantics](solar-wind-aggregation.md) define stored statistics and
recalculation. [Cleanup](solar-wind-cleanup.md) defaults to a report and requires
`--apply` for deletion. Archive backfill is not implemented.
