"""Product identities and adapter contracts; no network or database work on import."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import math
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class ObservationDefinition:
    unit: str
    resolution: str = '1h'
    kind: Literal['numeric', 'file'] = 'numeric'
    quality: str = 'finite_nonnegative'
    live_resolution: str = '1min'
    max_age: timedelta = timedelta(minutes=10)

    def accepts(self, value: float) -> bool:
        return math.isfinite(value) and (self.quality == 'finite' or value >= 0) and (
            self.quality != 'kp' or value <= 9)


OBSERVATIONS = {
    'bx': ObservationDefinition('nT', quality='finite'),
    'by': ObservationDefinition('nT', quality='finite'),
    'bz': ObservationDefinition('nT', quality='finite'),
    'v': ObservationDefinition('km/s'),
    'n': ObservationDefinition('cm^-3'),
    't': ObservationDefinition('K'),
    'kp': ObservationDefinition('index', resolution='3h', quality='kp', live_resolution='3h', max_age=timedelta(hours=7)),
    'ap': ObservationDefinition('nT', resolution='3h', live_resolution='3h', max_age=timedelta(hours=7)),
    'dst': ObservationDefinition('nT', quality='finite', live_resolution='1h', max_age=timedelta(hours=3)),
    'f10_7': ObservationDefinition('sfu', resolution='1D', live_resolution='1D', max_age=timedelta(days=2)),
    'goes': ObservationDefinition('JSON', kind='file', live_resolution='1h', max_age=timedelta(hours=2)),
    'gong': ObservationDefinition('FITS', kind='file', live_resolution='1h', max_age=timedelta(hours=3)),
}


@dataclass(frozen=True)
class Product:
    metrics: frozenset[str]
    modes: frozenset[str]
    kind: Literal['numeric', 'file'] = 'numeric'


# All numeric products feed measurement; live wind uses native minute RTSW.
PRODUCTS = {
    'omni.hourly': Product(frozenset({'bx', 'by', 'bz', 'v', 'n', 't', 'kp', 'ap', 'dst', 'f10_7'}),
                          frozenset({'historical'})),
    'soho.plasma_hourly': Product(frozenset({'v', 'n', 't'}), frozenset({'historical'})),
    'ace.plasma_hourly': Product(frozenset({'v', 'n', 't'}), frozenset({'historical'})),
    'swpc.rtsw_plasma': Product(frozenset({'v', 'n', 't'}), frozenset({'live'})),
    'swpc.rtsw_mag': Product(frozenset({'bx', 'by', 'bz'}), frozenset({'live'})),
    'swpc.kp': Product(frozenset({'kp', 'ap'}), frozenset({'live'})),
    'swpc.dst': Product(frozenset({'dst'}), frozenset({'live'})),
    'swpc.f107': Product(frozenset({'f10_7'}), frozenset({'live'})),
    'gfz.f107': Product(frozenset({'f10_7'}), frozenset({'historical'})),
    'goes.live': Product(frozenset({'goes'}), frozenset({'live'}), kind='file'),
    'goes.archive': Product(frozenset({'goes'}), frozenset({'historical'}), kind='file'),
    'gong.live': Product(frozenset({'gong'}), frozenset({'live'}), kind='file'),
    'gong.archive': Product(frozenset({'gong'}), frozenset({'historical'}), kind='file'),
}


class NumericAdapter(Protocol):
    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Return metric/value/observed_at records in the UTC range [start, end)."""
        ...


class FileAdapter(Protocol):
    def fetch(self, slot: datetime, root: Path, expected: dict | None = None) -> dict | None:
        """Return a file receipt; preserve originals and recorded availability."""
        ...
