"""Public quantile blending and application of fitted calibration parameters."""
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


QUANTILE_NAMES = ("q10", "q50", "q90")


def uses_overlapping_buckets(buckets: Sequence[tuple]) -> bool:
    """Return whether buckets are stored as numeric (start, end) ranges."""
    return bool(buckets) and all(
        len(bucket) == 2
        and isinstance(bucket[0], (int, float, np.number))
        and isinstance(bucket[1], (int, float, np.number))
        for bucket in buckets
    )


def validate_overlapping_buckets(
    buckets: Sequence[tuple[int, int]],
    lead_hours: Sequence[int] | np.ndarray,
) -> None:
    if not uses_overlapping_buckets(buckets):
        raise ValueError("Overlapping buckets must be numeric (start, end) pairs")

    for start, end in buckets:
        if start >= end:
            raise ValueError(f"Invalid lead bucket ({start}, {end})")

    leads = np.asarray(lead_hours, dtype=float)
    if leads.size == 0:
        return

    covered = np.zeros(leads.shape, dtype=bool)
    for start, end in buckets:
        covered |= (leads >= start) & (leads <= end)

    if not covered.all():
        missing = np.unique(leads[~covered]).tolist()
        raise ValueError(f"Lead buckets do not cover lead hours: {missing}")


def overlapping_bucket_weights(
    lead_hours: Sequence[int] | np.ndarray,
    buckets: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Build normalized raised-cosine weights for overlapping lead ranges."""
    leads = np.asarray(lead_hours, dtype=float)
    validate_overlapping_buckets(buckets, leads)

    weights = np.zeros((len(leads), len(buckets)), dtype=float)
    covering = np.zeros_like(weights, dtype=bool)

    for column, (start, end) in enumerate(buckets):
        mask = (leads >= start) & (leads <= end)
        position = (leads[mask] - start) / (end - start)
        covering[mask, column] = True
        weights[mask, column] = np.sin(np.pi * position) ** 2

    # At the global edges (and for buckets that only touch at an endpoint), all
    # raised-cosine weights can be zero. Use the closest covering bucket there.
    zero_rows = np.flatnonzero(weights.sum(axis=1) == 0.0)
    centers = np.asarray([(start + end) / 2 for start, end in buckets])
    for row in zero_rows:
        candidates = np.flatnonzero(covering[row])
        closest = candidates[np.argmin(np.abs(centers[candidates] - leads[row]))]
        weights[row, closest] = 1.0

    return weights / weights.sum(axis=1, keepdims=True)


def predict_overlapping_quantiles(
    frame: pd.DataFrame,
    models: Mapping[tuple[tuple[int, int], str], object],
    features: Sequence[str],
    buckets: Sequence[tuple[int, int]],
) -> pd.DataFrame:
    """Predict, order and smoothly blend q10/q50/q90 bucket models."""
    weights = overlapping_bucket_weights(frame["lead_hours"].to_numpy(), buckets)
    blended = np.zeros((len(frame), len(QUANTILE_NAMES)), dtype=float)

    for bucket_index, bucket in enumerate(buckets):
        bucket_weights = weights[:, bucket_index]
        mask = bucket_weights > 0.0
        if not mask.any():
            continue

        bucket_predictions = np.column_stack([
            models[(bucket, quantile)].predict(frame.loc[mask, features])
            for quantile in QUANTILE_NAMES
        ])
        bucket_predictions.sort(axis=1)
        blended[mask] += bucket_predictions * bucket_weights[mask, np.newaxis]

    return pd.DataFrame(blended, columns=QUANTILE_NAMES, index=frame.index)


def apply_quantile_calibration(
    predictions: pd.DataFrame,
    lead_hours: Sequence[int] | np.ndarray,
    calibration: pd.DataFrame,
) -> pd.DataFrame:
    """Apply lead-dependent calibration to q10/q50/q90 predictions."""
    parameters = calibration.set_index("lead_hours").reindex(
        np.asarray(lead_hours, dtype=int)
    )
    if parameters[["median_bias", "lower_scale", "upper_scale"]].isna().any().any():
        missing = np.unique(parameters.index[parameters["median_bias"].isna()]).tolist()
        raise ValueError(f"Missing calibration for lead hours: {missing}")

    q10 = predictions["q10"].to_numpy()
    q50 = predictions["q50"].to_numpy()
    q90 = predictions["q90"].to_numpy()
    median = q50 + parameters["median_bias"].to_numpy()

    calibrated = np.column_stack([
        median - parameters["lower_scale"].to_numpy() * (q50 - q10),
        median,
        median + parameters["upper_scale"].to_numpy() * (q90 - q50),
    ])
    calibrated.sort(axis=1)

    return pd.DataFrame(calibrated, columns=QUANTILE_NAMES, index=predictions.index)
