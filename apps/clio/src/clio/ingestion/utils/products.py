"""Product identities and adapter contracts; no network or database work on import."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import TYPE_CHECKING, Literal, Protocol

from common.schemas.sdo_images import OBSERVED_CHANNELS
from common.schemas.sdo_browse import CHANNELS as BROWSE_CHANNELS
from common.observationcatalog import METRICS

if TYPE_CHECKING:
    import pandas as pd

XRS_METRICS = ('xrs_a', 'xrs_b')
PROTON_METRICS = ('proton_10mev', 'proton_100mev')
WIND_METRICS = ('bx', 'by', 'bz', 'v', 'n', 't')


@dataclass(frozen=True)
class ObservationDefinition:
    unit: str
    kind: Literal['numeric', 'file'] = 'numeric'
    signed: bool = False
    max_value: float | None = None
    coverage: Literal['slots', 'latest'] = 'slots'
    channel: str | None = None
    file_kind: str | None = None

    def accepts(self, value: float) -> bool:
        return math.isfinite(value) and (self.signed or value >= 0) and (
            self.max_value is None or value <= self.max_value)


OBSERVATIONS = {
    **{
        name: ObservationDefinition(
            entry['unit'], signed=entry.get('signed', False), max_value=entry.get('max'),
            coverage='latest' if entry['group'] == 'model' else 'slots')
        for name, entry in METRICS.items() if name not in ('s10', 'm10', 'y10')
    },
    'goes': ObservationDefinition('JSON', kind='file'),
    'gong': ObservationDefinition('FITS', kind='file'),
}


@dataclass(frozen=True)
class Product:
    metrics: frozenset[str]
    modes: frozenset[str]
    kind: Literal['numeric', 'file'] = 'numeric'
    slot_minutes: int | None = None
    lock: int | None = None


# All numeric products feed measurement; live wind uses native minute RTSW.
PRODUCTS = {
    'omni.hourly': Product(frozenset({*WIND_METRICS, 'bt', 'kp', 'ap', 'dst', 'f10_7'}),
                          frozenset({'historical'}), slot_minutes=60),
    'soho.plasma_hourly': Product(frozenset({'v', 'n', 't'}), frozenset({'historical'}), slot_minutes=60),
    'ace.plasma_hourly': Product(frozenset({'v', 'n', 't'}), frozenset({'historical'}), slot_minutes=60),
    'swpc.rtsw_plasma': Product(frozenset({'v', 'n', 't'}), frozenset({'live'}), slot_minutes=1, lock=730101),
    'swpc.rtsw_mag': Product(frozenset({'bx', 'by', 'bz', 'bt'}), frozenset({'live'}), slot_minutes=1, lock=730100),
    'swpc.kp': Product(frozenset({'kp', 'ap'}), frozenset({'live'}), slot_minutes=180, lock=730110),
    'swpc.dst': Product(frozenset({'dst'}), frozenset({'live'}), slot_minutes=60, lock=730111),
    'swpc.f107': Product(frozenset({'f10_7'}), frozenset({'live'})),
    'gfz.f107': Product(frozenset({'f10_7'}), frozenset({'historical'})),
    'celestrak.f107': Product(frozenset({'f10_7', 'f10_7_81mean'}), frozenset({'live'})),
    'celestrak.f107_archive': Product(frozenset({'f10_7', 'f10_7_81mean'}), frozenset({'historical'})),
    'solfsmy.means': Product(frozenset({'f10_7_81mean', 's10_81mean', 'm10_81mean', 'y10_81mean'}), frozenset({'live'})),
    'solfsmy.means_archive': Product(frozenset({'f10_7_81mean', 's10_81mean', 'm10_81mean', 'y10_81mean'}), frozenset({'historical'})),
    'goes.xrs': Product(frozenset(XRS_METRICS), frozenset({'live', 'historical'}), slot_minutes=1, lock=730120),
    'goes.protons': Product(frozenset(PROTON_METRICS), frozenset({'live', 'historical'}), slot_minutes=5, lock=730121),
    'goes.live': Product(frozenset({'goes'}), frozenset({'live'}), kind='file', slot_minutes=60),
    'goes.archive': Product(frozenset({'goes'}), frozenset({'historical'}), kind='file', slot_minutes=1440),
    'gong.live': Product(frozenset({'gong'}), frozenset({'live'}), kind='file', slot_minutes=60),
    'gong.archive': Product(frozenset({'gong'}), frozenset({'historical'}), kind='file', slot_minutes=60),
}


for channel in OBSERVED_CHANNELS:
    name = f'aia_{int(channel[3:]):03d}_nrt' if channel.startswith('aia') else 'hmi_m_nrt'
    source = 'aia.nrt_' + channel[3:] if channel.startswith('aia') else 'hmi.nrt_m'
    OBSERVATIONS[name] = ObservationDefinition('FITS', kind='file',
        channel=channel, file_kind='sdo')
    PRODUCTS[source] = Product(frozenset({name}), frozenset({'live', 'historical'}), kind='file', slot_minutes=60)

browse_metrics = set()
for channel in BROWSE_CHANNELS:
    name = f'aia_{channel}_browse'
    OBSERVATIONS[name] = ObservationDefinition('JPEG', kind='file', coverage='latest',
        channel='aia' + str(int(channel)), file_kind='sdo_browse')
    browse_metrics.add(name)
PRODUCTS['nasa.aia_browse'] = Product(frozenset(browse_metrics), frozenset({'live'}), kind='file')

FILE_OBSERVATIONS = {(definition.file_kind or name, definition.channel): name
                     for name, definition in OBSERVATIONS.items() if definition.kind == 'file'}


class NumericAdapter(Protocol):
    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Return metric/value/observed_at records in the UTC range [start, end)."""
        ...
