"""Metrics stored in hourly observations."""
REQUIRED_METRICS = ('bx', 'by', 'bz', 'v', 'n', 't', 'kp', 'dst', 'ap', 'f10_7')
SOLAR_INDEX_METRICS = ('s10', 'm10', 'y10')
OBSERVATION_METRICS = (*REQUIRED_METRICS, *SOLAR_INDEX_METRICS)
