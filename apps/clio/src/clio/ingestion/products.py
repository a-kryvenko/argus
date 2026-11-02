"""Product identities and adapter contracts; no network or database work on import."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
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
    signed: bool = False
    max_value: float | None = None
    live_resolution: str = '1min'
    coverage: Literal['slots', 'latest'] = 'slots'

    def accepts(self, value: float) -> bool:
        return math.isfinite(value) and (self.signed or value >= 0) and (
            self.max_value is None or value <= self.max_value)


OBSERVATIONS = {
    'bx': ObservationDefinition('nT', signed=True),
    'by': ObservationDefinition('nT', signed=True),
    'bz': ObservationDefinition('nT', signed=True),
    'v': ObservationDefinition('km/s'),
    'n': ObservationDefinition('cm^-3'),
    't': ObservationDefinition('K'),
    'kp': ObservationDefinition('index', resolution='3h', max_value=9, live_resolution='3h'),
    'ap': ObservationDefinition('nT', resolution='3h', live_resolution='3h'),
    'dst': ObservationDefinition('nT', signed=True, live_resolution='1h'),
    'f10_7': ObservationDefinition('sfu', resolution='1D', live_resolution='1D'),
    'f10_7_81mean': ObservationDefinition('sfu', resolution='1D', live_resolution='1D', coverage='latest'),
    **{name + '_81mean': ObservationDefinition('sfu', resolution='1D', live_resolution='1D', coverage='latest')
       for name in ('s10', 'm10', 'y10')},
    'goes': ObservationDefinition('JSON', kind='file', live_resolution='1h'),
    'gong': ObservationDefinition('FITS', kind='file', live_resolution='1h'),
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
    'celestrak.f107': Product(frozenset({'f10_7', 'f10_7_81mean'}), frozenset({'live'})),
    'celestrak.f107_archive': Product(frozenset({'f10_7', 'f10_7_81mean'}), frozenset({'historical'})),
    'solfsmy.means': Product(frozenset({'f10_7_81mean', 's10_81mean', 'm10_81mean', 'y10_81mean'}), frozenset({'live'})),
    'solfsmy.means_archive': Product(frozenset({'f10_7_81mean', 's10_81mean', 'm10_81mean', 'y10_81mean'}), frozenset({'historical'})),
    'goes.live': Product(frozenset({'goes'}), frozenset({'live'}), kind='file'),
    'goes.archive': Product(frozenset({'goes'}), frozenset({'historical'}), kind='file'),
    'gong.live': Product(frozenset({'gong'}), frozenset({'live'}), kind='file'),
    'gong.archive': Product(frozenset({'gong'}), frozenset({'historical'}), kind='file'),
}


SOURCE_IDS = {
    'swpc.rtsw_mag': 'solar_wind_mag',
    'swpc.rtsw_plasma': 'solar_wind_plasma',
    'swpc.kp': 'kp',
    'swpc.dst': 'dst',
}


class NumericAdapter(Protocol):
    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Return metric/value/observed_at records in the UTC range [start, end)."""
        ...


class FileAdapter(Protocol):
    def fetch(self, slot: datetime, root: Path, expected: dict | None = None) -> dict | None:
        """Return a file receipt; preserve originals and recorded availability."""
        ...
