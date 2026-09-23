"""Exploratory AIA 193 dark-region features; no magnetic CH classification.

Use level-1.5 synoptic *snapshots*, not completed Carrington synoptic maps.
All extraction uses one exposure; temporal comparisons only use older frames.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import ndimage

VERSION = 1
LAT = np.arange(-60., 61., 2.)
LON = np.arange(-60., 61., 2.)
LON_GRID, LAT_GRID = np.meshgrid(LON, LAT)
AREA = np.cos(np.deg2rad(LAT_GRID))
ROTATION_HOURS = 27.2753 * 24
LAT_EDGES = [-61, -20, 20, 61]
LON_EDGES = [-61, -36, -12, 12, 36, 61]


def sectors():
    for i, (a, b) in enumerate(zip(LAT_EDGES[:-1], LAT_EDGES[1:])):
        for j, (c, d) in enumerate(zip(LON_EDGES[:-1], LON_EDGES[1:])):
            yield f"lat{i}_lon{j}", ((LAT_GRID >= a) & (LAT_GRID < b)
                                      & (LON_GRID >= c) & (LON_GRID < d))


def weighted_mean(values, mask):
    good = mask & np.isfinite(values)
    return float(np.average(values[good], weights=AREA[good])) if good.any() else np.nan


def segment_dark(intensity, mu, threshold=0.45, min_pixels=4):
    """Per-image radial median correction, then threshold and component filter.

    Scalar exposure/degradation factors cancel in the ratio. This is NOT a
    full instrument calibration or a validated coronal-hole segmentation.
    """
    if not 0 < threshold < 1 or min_pixels < 1:
        raise ValueError("Invalid segmentation parameters")
    valid = np.isfinite(intensity) & (intensity > 0) & (mu >= 0.35)
    ratio = np.full(intensity.shape, np.nan)
    for low, high in zip(np.linspace(.35, 1.001, 9)[:-1], np.linspace(.35, 1.001, 9)[1:]):
        ring = valid & (mu >= low) & (mu < high)
        if ring.any():
            ratio[ring] = intensity[ring] / np.median(intensity[ring])
    labels, _ = ndimage.label(np.isfinite(ratio) & (ratio < threshold))
    sizes = np.bincount(labels.ravel())
    dark = (labels > 0) & (sizes[labels] >= min_pixels)
    return np.where(np.isfinite(ratio), dark.astype(float), np.nan), ratio


def extract_frame(path, *, threshold=0.45, min_pixels=4):
    """Map FITS WCS to heliographic coordinates, including roll and B0 tilt."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    import sunpy.map
    from sunpy.coordinates import frames

    path = Path(path)
    smap = sunpy.map.Map(path)
    if smap.wavelength.to_value(u.angstrom) != 193:
        raise ValueError("Expected AIA 193 Angstrom")
    if smap.processing_level != 1.5:
        raise ValueError("Use level-1.5 synoptic snapshots; raw level-1 needs calibration")
    if int(smap.meta.get("quality", -1)) != 0:
        raise ValueError("Missing/nonzero FITS QUALITY")
    exposure = smap.exposure_time.to_value(u.s)
    if not np.isfinite(exposure) or exposure <= 0:
        raise ValueError("Invalid exposure time")
    coords = SkyCoord(LON_GRID*u.deg + smap.heliographic_longitude,
                      LAT_GRID*u.deg, smap.rsun_meters,
                      frame=frames.HeliographicStonyhurst, obstime=smap.date)
    px, py = smap.world_to_pixel(coords)
    intensity = ndimage.map_coordinates(smap.data.astype(float)/exposure,
        [py.to_value(u.pix), px.to_value(u.pix)], order=1, mode="constant", cval=np.nan)
    # Surface normal to observer, including the observer's changing latitude.
    b0 = smap.heliographic_latitude.to_value(u.rad)
    mu = (np.sin(np.deg2rad(LAT_GRID))*np.sin(b0)
          + np.cos(np.deg2rad(LAT_GRID))*np.cos(b0)*np.cos(np.deg2rad(LON_GRID)))
    eligible = mu >= .35
    valid_fraction = float((np.isfinite(intensity[eligible]) & (intensity[eligible] > 0)).mean())
    if valid_fraction < .98:
        raise ValueError(f"Insufficient valid disk pixels: {valid_fraction:.3f}")
    dark, ratio = segment_dark(intensity, mu, threshold, min_pixels)
    meta = dict(observed_at=pd.Timestamp(smap.date.utc.datetime).tz_localize("UTC").isoformat(),
                carrington_lon=float(smap.carrington_longitude.to_value(u.deg)),
                b0_deg=float(np.rad2deg(b0)), valid_fraction=valid_fraction,
                path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                threshold=threshold, min_pixels=min_pixels, version=VERSION)
    return dark.astype(np.float32), ratio.astype(np.float32), meta


def aligned_change(current, previous, current_lon, previous_lon):
    """Compare the same Carrington locations, only in the observed overlap.

    Carrington registration is a rigid-rotation approximation; it does not
    assume every structure follows differential rotation or survives a month.
    """
    shift = (current_lon - previous_lon + 180) % 360 - 180
    x = (LON_GRID + shift - LON[0]) / (LON[1]-LON[0])
    y = (LAT_GRID - LAT[0]) / (LAT[1]-LAT[0])
    old = ndimage.map_coordinates(previous, [y, x], order=0,
                                  mode="constant", cval=np.nan, prefilter=False)
    return current-old


def build_features(raw_dir, output_dir, *, assumed_latency_hours=2., threshold=.45,
                   min_pixels=4, pair_tolerance_hours=6.):
    """Incremental extraction with rejection audit and reproducible provenance.

    Archive headers do not give historical publication times. Availability is
    explicitly simulated as observation + latency, NOT claimed operational.
    """
    if assumed_latency_hours < 0 or not 0 < pair_tolerance_hours < 24:
        raise ValueError("Invalid latency/pair tolerance")
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir / "frames"
    cache.mkdir(exist_ok=True)
    protocol = dict(version=VERSION, threshold=threshold, min_pixels=min_pixels,
        assumed_latency_hours=assumed_latency_hours, pair_tolerance_hours=pair_tolerance_hours,
        latitude=LAT.tolist(), longitude=LON.tolist(), rotation_hours=ROTATION_HOURS,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    settings_path = output_dir / "extraction.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != protocol:
        raise ValueError("Extraction settings changed: choose a new output directory")
    settings_path.write_text(json.dumps(protocol, indent=2))
    files = sorted(raw_dir.rglob("AIA*_0193.fits"))
    if not files:
        raise FileNotFoundError(f"No synoptic AIA 193 snapshots under {raw_dir}")
    metadata, rejected = [], []
    for i, path in enumerate(files):
        destination = cache / (path.stem + ".npz")
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if destination.exists():
                with np.load(destination, allow_pickle=False) as saved:
                    meta = json.loads(str(saved["metadata"]))
                if meta["sha256"] != digest:
                    raise ValueError("Cached FITS changed; remove its cached frame before rebuilding")
            else:
                dark, ratio, meta = extract_frame(path, threshold=threshold, min_pixels=min_pixels)
                temporary = destination.with_suffix(".part")
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, dark=dark, ratio=ratio, metadata=json.dumps(meta))
                temporary.replace(destination)
            metadata.append({**meta, "cache_path": str(destination.resolve())})
        except (ValueError, OSError) as exc:
            rejected.append(dict(path=str(path), reason=str(exc)))
        if (i+1) % 100 == 0:
            print(f"AIA extraction: {i+1}/{len(files)}, rejected {len(rejected)}", flush=True)
    pd.DataFrame(rejected, columns=["path", "reason"]).to_csv(output_dir/"rejected.csv", index=False)
    if not metadata:
        raise ValueError("All AIA frames rejected; see rejected.csv")
    metadata.sort(key=lambda row: row["observed_at"])
    times = pd.DatetimeIndex([row["observed_at"] for row in metadata])
    if times.has_duplicates:
        raise ValueError("Duplicate AIA observation times")
    rows = []
    for i, meta in enumerate(metadata):
        with np.load(meta["cache_path"], allow_pickle=False) as saved:
            current = saved["dark"]
        row = dict(observed_at=times[i],
            available_at=times[i]+pd.Timedelta(hours=assumed_latency_hours),
            aia_valid_fraction=meta["valid_fraction"], aia_b0_deg=meta["b0_deg"])
        for name, mask in sectors():
            row[f"aia_area_{name}"] = weighted_mean(current, mask)
        for label, hours in [("24h", 24.), ("rotation", ROTATION_HOURS)]:
            target = times[i]-pd.Timedelta(hours=hours)
            k = int(times.searchsorted(target))
            candidates = [j for j in [k-1, k] if 0 <= j < i]
            j = min(candidates, key=lambda j: abs(times[j]-target)) if candidates else None
            change = np.full(current.shape, np.nan)
            row[f"aia_{label}_separation_h"] = np.nan
            if j is not None and abs(times[j]-target) <= pd.Timedelta(hours=pair_tolerance_hours):
                with np.load(metadata[j]["cache_path"], allow_pickle=False) as saved:
                    change = aligned_change(current, saved["dark"], meta["carrington_lon"],
                                            metadata[j]["carrington_lon"])
                row[f"aia_{label}_separation_h"] = (times[i]-times[j]).total_seconds()/3600
            for name, mask in sectors():
                row[f"aia_delta_{label}_{name}"] = weighted_mean(change, mask)
                row[f"aia_overlap_{label}_{name}"] = weighted_mean(np.isfinite(change).astype(float), mask)
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_parquet(output_dir/"features.parquet", index=False)
    pd.DataFrame(metadata).to_parquet(output_dir/"manifest.parquet", index=False)
    coverage = result.assign(year=result.observed_at.dt.year).groupby("year").agg(
        frames=("observed_at", "size"), first=("observed_at", "min"), last=("observed_at", "max"),
        rotation_pairs=("aia_rotation_separation_h", "count"))
    coverage.to_csv(output_dir/"coverage.csv")
    return result


def attach_features(issues, features, *, max_age_hours=12.):
    """Backward as-of by AVAILABLE time; never interpolate/backfill observations."""
    if max_age_hours <= 0:
        raise ValueError("max_age_hours must be positive")
    left = pd.DataFrame({"issue_time": pd.to_datetime(pd.Series(issues).unique(), utc=True)}).sort_values("issue_time")
    right = features.copy()
    for column in ["observed_at", "available_at"]:
        right[column] = pd.to_datetime(right[column], utc=True)
    if (right.available_at < right.observed_at).any() or right.available_at.duplicated().any():
        raise ValueError("Invalid or duplicated availability timestamps")
    right = right.sort_values("available_at")
    result = pd.merge_asof(left, right, left_on="issue_time", right_on="available_at", direction="backward")
    age = (result.issue_time-result.observed_at).dt.total_seconds()/3600
    stale = age.isna() | (age > max_age_hours)
    columns = [c for c in features if c.startswith("aia_")]
    result.loc[stale, columns] = np.nan
    result["aia_age_hours"] = age.where(~stale)
    result["aia_available"] = (~stale).astype(float)
    return result
