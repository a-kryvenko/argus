"""Observation storage and read failures."""


class ObservationUnavailable(RuntimeError):
    pass


class ObservationNotFound(LookupError):
    pass
