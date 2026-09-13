"""Versioned Prophet read contract; independent of storage and HTTP frameworks."""
import csv
import hashlib
import io
import math
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, Field, model_validator

# A release includes all sources needed to render one public/private product.
PRODUCT_ARTIFACTS = {
    'solar-wind-speed': ('plasma_speed_quantile', 'plasma_speed_threshold'),
    'solar-wind-density': ('plasma_density_quantile',),
    'hmf': ('hmf_total_threshold', 'hmf_southward_threshold'),
    'solar-radiation': ('f10_7_quantile', 's10_quantile', 'm10_quantile', 'y10_quantile'),
    'geomagnetic-activity': ('kp_threshold', 'ap_quantile'),
    'dst': ('dst_quantile',),
    'atmospheric-density': ('atmospheric_density',),
}
PREDICTION_COLUMNS = {
    **{name: tuple(f'{prefix}_q{q}' for q in (10, 50, 90)) for name, prefix in (
        ('plasma_speed_quantile', 'v'), ('plasma_density_quantile', 'n'),
        ('ap_quantile', 'ap'), ('dst_quantile', 'dst'), ('f10_7_quantile', 'f107'),
        ('s10_quantile', 's10'), ('m10_quantile', 'm10'), ('y10_quantile', 'y10'))},
    **{name: tuple(f'p_{prefix}_ge_{threshold}' for threshold in thresholds) for name, prefix, thresholds in (
        ('plasma_speed_threshold', 'v', (450, 500, 600)), ('kp_threshold', 'kp', (4, 5, 6)),
        ('hmf_total_threshold', 'bt', (5, 10, 15)), ('hmf_southward_threshold', 'bs', (5, 10, 15)))},
    'atmospheric_density': ('altitude_km', 'latitude_deg', 'rho_kg_m3', 'rho_lon_p10_kg_m3', 'rho_lon_p90_kg_m3'),
}
MAX_ARTIFACT_BYTES = 32 * 1024 * 1024


class ForecastArtifact(BaseModel):
    name: str
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    row_count: int = Field(gt=0)
    columns: list[str]
    csv_text: str = Field(max_length=MAX_ARTIFACT_BYTES)
    model_info: dict

    def issue_time(self) -> datetime:
        reader = csv.DictReader(io.StringIO(self.csv_text))
        if reader.fieldnames != self.columns or len(set(self.columns)) != len(self.columns):
            raise ValueError('Artifact columns do not match')
        if not {'issue_time', 'valid_time', 'lead_hours'}.issubset(self.columns):
            raise ValueError('Artifact time columns are missing')
        predictions = PREDICTION_COLUMNS.get(self.name)
        if predictions is None or not set(predictions).issubset(self.columns):
            raise ValueError('Artifact prediction columns are missing')
        issue = None
        count = 0
        for row in reader:
            if None in row or None in row.values():
                raise ValueError('Malformed CSV row')
            for column in predictions:
                value = float(row[column])
                if not math.isfinite(value) or (column.startswith('p_') and not 0 <= value <= 1):
                    raise ValueError('Invalid forecast prediction')
            current = datetime.fromisoformat(row['issue_time'])
            valid = datetime.fromisoformat(row['valid_time'])
            lead = int(row['lead_hours'])
            if current.tzinfo is None or valid.tzinfo is None or lead < 0:
                raise ValueError('Artifact times must be timezone-aware with nonnegative leads')
            if issue is not None and issue != current:
                raise ValueError('Mixed issue times in artifact')
            issue = current
            count += 1
        if count != self.row_count:
            raise ValueError('Artifact row count does not match')
        return issue

    @model_validator(mode='after')
    def check_content(self):
        content = self.csv_text.encode('utf-8')
        if len(content) > MAX_ARTIFACT_BYTES or hashlib.sha256(content).hexdigest() != self.sha256:
            raise ValueError('Artifact checksum or size does not match')
        self.issue_time()
        return self


class ForecastRelease(BaseModel):
    contract_version: Literal[1] = 1
    release_id: UUID
    run_id: UUID
    product: str
    published_at: AwareDatetime
    issue_time: AwareDatetime
    artifacts: list[ForecastArtifact]

    @model_validator(mode='after')
    def check_product(self):
        names = [artifact.name for artifact in self.artifacts]
        expected = PRODUCT_ARTIFACTS.get(self.product)
        if expected is None or len(names) != len(expected) or set(names) != set(expected):
            raise ValueError('Incomplete product release')
        if any(artifact.issue_time() != self.issue_time for artifact in self.artifacts):
            raise ValueError('Mixed issue times in release')
        return self
