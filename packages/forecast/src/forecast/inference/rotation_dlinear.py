"""Speed-only rotation DLinear inference without a PyTorch runtime."""
from pathlib import Path

import numpy as np
import pandas as pd
from common.data.omni import OMNI_FILL_VALUES


class RotationDLinearForecaster:
    """Predict speed from hourly historical windows using a fitted linear bundle.

    The bundle contains folded DLinear weights, normalization and history offsets.
    ``add_rotation_v`` preserves request order and index; missing histories yield
    NaN. It never reads targets. Observation timestamps denote availability time.
    """

    registry_name = "plasma_speed_dlinear"
    artifact_format = "rotation_dlinear"
    variable = "v"
    history_name = "speed"

    def __init__(self, bundle, *, batch_size=256):
        if bundle.get("format") != self.artifact_format or bundle.get("version") != 1:
            raise ValueError("Unsupported rotation DLinear artifact format")
        self.settings = dict(bundle["settings"])
        settings = self.settings
        if settings["columns"] != [self.variable]:
            raise ValueError(f"Rotation DLinear requires exactly one input: {self.variable}")
        segments = settings["segments"]["rotations"]
        if (not segments or any(len(s) != 2 for s in segments)
                or any(not isinstance(v, int) for s in segments for v in s)
                or any(a > b or b > 0 for a, b in segments)):
            raise ValueError("History segments must have integer hourly bounds at or before issue_time")
        if any(segments[i][1] >= segments[i + 1][0] for i in range(len(segments) - 1)):
            raise ValueError("History segments must be ordered and disjoint")
        self.offsets = np.concatenate([np.arange(a, b + 1) for a, b in segments])
        self.horizon = int(settings["horizon"])
        self.ffill_limit = int(settings["ffill_limit_hours"])
        self.mean = np.float32(settings["mean"][0])
        self.std = np.float32(settings["std"][0])
        if (self.horizon < 1 or self.ffill_limit < 1 or not isinstance(batch_size, int)
                or batch_size < 1 or not np.isfinite([self.mean, self.std]).all() or self.std <= 0):
            raise ValueError("Invalid rotation DLinear configuration")
        self.batch_size = batch_size
        self.weights = np.array(bundle["weights"], dtype=np.float64, copy=True)
        self.bias = np.array(bundle["bias"], dtype=np.float64, copy=True)
        if self.weights.shape != (self.horizon, len(self.offsets)) or self.bias.shape != (self.horizon,):
            raise ValueError("Weights do not match the configured history and horizon")
        if not np.isfinite(self.weights).all() or not np.isfinite(self.bias).all():
            raise ValueError("Model weights must be finite")

    @classmethod
    def load(cls, path, *, batch_size=256):
        """Load a trusted joblib bundle; joblib is an optional model dependency."""
        import joblib
        return cls(joblib.load(Path(path)), batch_size=batch_size)

    @classmethod
    def from_registry(cls, *, workdir, registry, batch_size=256):
        """Resolve an entry from the models mapping in models_registry.yaml."""
        entry = registry[cls.registry_name]
        return cls.load(Path(workdir) / entry["artifact_path"], batch_size=batch_size)

    @staticmethod
    def _times(values):
        times = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
        if times.hasnans or not (times == times.floor("h")).all():
            raise ValueError("issue_time must contain nonmissing hourly timestamps")
        return times

    def add_rotation_v(self, frame, observations, *, column="rotation_v", require_history=False):
        """Return a copy of frame with one forecast per (issue_time, lead_hours)."""
        return self._add_predictions(frame, observations, column=column, require_history=require_history)

    def _clean_values(self, values):
        return values.where(values.ge(0) & values.lt(OMNI_FILL_VALUES[self.variable]))

    def _add_predictions(self, frame, observations, *, column, require_history=False):
        result = frame.copy()
        issue_times = self._times(frame["issue_time"])
        leads = pd.to_numeric(frame["lead_hours"], errors="raise").to_numpy(dtype=float)
        if (not np.isfinite(leads).all() or (leads != np.floor(leads)).any()
                or (leads < 1).any() or (leads > self.horizon).any()):
            raise ValueError(f"lead_hours must be integers in 1..{self.horizon}")
        result[column] = np.nan
        if frame.empty or observations.empty:
            if require_history and not frame.empty:
                missing = issue_times[0] + pd.to_timedelta(self.offsets, unit="h")
                raise self._history_error(issue_times[0], missing)
            return result
        obs = pd.DataFrame({"issue_time": self._times(observations["issue_time"]),
                            self.variable: observations[self.variable].to_numpy(dtype=np.float64)})
        # Historical databases can retain provider fill values. Treat them as
        # gaps before normalization and the existing bounded forward-fill.
        obs[self.variable] = self._clean_values(obs[self.variable])
        if (obs.groupby("issue_time")[self.variable].nunique(dropna=False) > 1).any():
            raise ValueError(f"Conflicting {self.variable} observations at the same issue_time")
        series = obs.drop_duplicates("issue_time").set_index("issue_time")[self.variable].sort_index()
        # Include a fill buffer before the first requested historical point.
        start = issue_times.min() + pd.Timedelta(int(self.offsets.min()) - self.ffill_limit, unit="h")
        clock = pd.date_range(start, issue_times.max(), freq="h")
        history = series.reindex(clock).ffill(limit=self.ffill_limit).to_numpy(dtype=np.float32)
        history = (history - self.mean) / self.std
        codes, unique_times = pd.factorize(issue_times, sort=True)
        origins = clock.get_indexer(unique_times)
        forecasts = np.full((len(unique_times), self.horizon), np.nan, dtype=np.float64)
        for begin in range(0, len(unique_times), self.batch_size):
            end = min(begin + self.batch_size, len(unique_times))
            indices = origins[begin:end, None] + self.offsets
            windows = history[indices]
            valid = np.isfinite(windows).all(axis=1)
            if require_history and not valid.all():
                row = np.flatnonzero(~valid)[0]
                missing = clock[indices[row][~np.isfinite(windows[row])]]
                raise self._history_error(unique_times[begin + row], missing)
            if valid.any():
                pred = (windows[valid] @ self.weights.T + self.bias) * self.std + self.mean
                if not np.isfinite(pred).all():
                    raise ValueError("DLinear returned nonfinite predictions")
                forecasts[begin + np.flatnonzero(valid)] = pred
        result[column] = forecasts[codes, leads.astype(int) - 1]
        return result

    def _history_error(self, issue, missing):
        """Bounded diagnostic using the exact post-fill inference window."""
        groups = np.split(missing, np.flatnonzero(np.diff(missing.asi8) != pd.Timedelta(1, unit="h").value) + 1)
        gaps = '; '.join(f'{group[0].isoformat()}..{group[-1].isoformat()} ({len(group)} h)'
                         for group in groups[:5])
        if len(groups) > 5:
            gaps += f'; ... {len(groups) - 5} more intervals'
        windows = '; '.join(
            f'{(issue + pd.Timedelta(start, unit="h")).isoformat()}..{(issue + pd.Timedelta(end, unit="h")).isoformat()}'
            for start, end in self.settings['segments']['rotations'])
        return ValueError(
            f'Insufficient hourly {self.history_name} history for DLinear at {issue.isoformat()}: '
            f'{len(missing)}/{len(self.offsets)} required hours remain missing after '
            f'forward-fill limited to {self.ffill_limit} h. Missing intervals (UTC, inclusive): {gaps}. '
            f'Required windows (UTC, inclusive): {windows}. '
            f'Restore observed {self.history_name} in these intervals before retrying; '
            'normalized/interpolated observations do not replace this history.'
        )
