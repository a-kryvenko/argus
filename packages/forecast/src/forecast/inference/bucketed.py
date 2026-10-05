"""Shared preparation of estimator bundles grouped by forecast horizon."""
import pandas as pd

from forecast.inference.base import DefaultForecastService
from forecast.inference.feature_models import embedded_features


class BucketedForecastService(DefaultForecastService):
    def __init__(self, models_bundle):
        super().__init__(models_bundle, feature_models=embedded_features(models_bundle))

    def _apply_lead_buckets(self, df, lead_buckets):
        from forecast.inference.quantiles import uses_overlapping_buckets

        if uses_overlapping_buckets(lead_buckets):
            return

        bins = [0] + [upper for upper, _ in lead_buckets]
        labels = [label for _, label in lead_buckets]

        df["lead_bucket"] = pd.cut(
            df["lead_hours"],
            bins=bins,
            labels=labels,
            include_lowest=True,
        )
