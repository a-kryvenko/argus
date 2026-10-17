# Atmospheric density advisories

`/dashboard/risk/leo` monitors the altitude/latitude envelope defined by:

- Name, apoapsis/periapsis altitude (90–2,500 km, apoapsis ≥ periapsis), inclination (0–180°).
- Optional positive weight (kg) and effective area (m²); stored as metadata.
- Review/high density-ratio thresholds. All saved spacecraft are monitored.

No trajectory upload is required. Each forecast time reports the largest
model density/background ratio across periapsis–apoapsis and latitudes
±min(inclination, 180°−inclination). This rectangular envelope can include
combinations absent from the actual orbit; it does not establish spacecraft
position or exposure. Missing grid coverage remains explicit.

The reference is Prophet's published fixed-geomagnetic model background.
Missing background leaves ratios unassessed. Density is longitude-averaged;
CME arrival context is separate and does not modify density. Thresholds are
attention rules, not calibrated failure probabilities.

Saved runs retain source releases and profile revisions. Journal entries link
immutable source runs; acknowledgment records review only. Older trajectory
profiles/history remain readable; saving a profile switches it to orbital parameters.

API: `/api/v1/dashboard/density-advisory` (direct Uvicorn: omit `/api/v1`).
Session-owned CRUD/assessment/history/journal routes; writes require dashboard Origin.
`DELETE /{id}?revision=N` removes the spacecraft, saved runs and journal; shared releases remain retained.
Worker polls every 60 seconds. [Operation](../apps/intelligence/README.md).
