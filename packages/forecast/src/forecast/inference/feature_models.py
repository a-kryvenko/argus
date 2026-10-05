"""Embedded model features selected by the bucketed bundle adapter."""
import pandas as pd


class EmbeddedSpeedFeature:
    def __init__(self, dependency):
        from forecast.inference.rotation_dlinear import RotationDLinearForecaster

        if dependency is None or 'bundle' not in dependency:
            raise ValueError('Models using dlinear_v must embed their fitted DLinear bundle')
        self.model = RotationDLinearForecaster(dependency['bundle'])

    def snapshot_options(self, inputs):
        return {'speed_history': pd.DataFrame(
            [point.model_dump() for point in inputs.speed_observations], columns=['issue_time', 'v'])}

    def prepare(self, observations, *, issue_time, speed_history=None):
        source = observations if speed_history is None else speed_history
        history = source[['issue_time', 'v']].copy()
        history['issue_time'] = pd.to_datetime(history.issue_time, utc=True)
        return history.loc[history.issue_time <= issue_time]

    def apply(self, frame, history, *, issue_time):
        requests = pd.DataFrame({'issue_time': [issue_time] * len(frame),
                                 'lead_hours': frame['lead_hours'].to_numpy()})
        predictions = self.model.add_rotation_v(
            requests, history, column='dlinear_v', require_history=True)
        frame['dlinear_v'] = predictions.dlinear_v.to_numpy()


def embedded_features(bundle):
    if 'dlinear_v' not in bundle.get('feature_columns', []):
        return ()
    return (EmbeddedSpeedFeature(bundle.get('feature_models', {}).get('dlinear_v')),)
