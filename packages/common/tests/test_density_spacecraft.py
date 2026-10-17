from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from common.schemas.density_advisory import AdvisoryProfile, AdvisorySpacecraftInput


def payload():
    return dict(name='Satellite A', apoapsis_km=420, periapsis_km=400,
                inclination_deg=51.6)


@pytest.mark.parametrize('field', ['name', 'apoapsis_km', 'periapsis_km', 'inclination_deg'])
def test_writes_require_spacecraft_fields(field):
    body = payload()
    del body[field]
    with pytest.raises(ValidationError):
        AdvisorySpacecraftInput.model_validate(body)


@pytest.mark.parametrize('changes', [
    {'name': ' '}, {'apoapsis_km': 399}, {'periapsis_km': 89},
    {'apoapsis_km': 2501}, {'inclination_deg': -1}, {'inclination_deg': 181},
    {'weight_kg': 0}, {'weight_kg': -1}, {'effective_area_m2': 0},
    {'effective_area_m2': -1}, {'apoapsis_km': float('nan')},
    {'weight_kg': float('inf')},
])
def test_invalid_spacecraft_values(changes):
    with pytest.raises(ValidationError):
        AdvisorySpacecraftInput.model_validate(payload() | changes)


def test_optional_properties_and_circular_equatorial_orbit_round_trip():
    body = payload() | dict(apoapsis_km=400, inclination_deg=0)
    profile = AdvisorySpacecraftInput.model_validate(body)
    assert profile.trajectory is None and profile.reference is None
    assert profile.reference_mode == 'prophet_background'
    assert 'max_sample_gap_minutes' not in profile.policy.model_dump()
    assert profile.weight_kg is None
    assert profile.effective_area_m2 is None
    profile = AdvisorySpacecraftInput.model_validate(body | dict(weight_kg=100, effective_area_m2=1.5))
    assert AdvisorySpacecraftInput.model_validate_json(profile.model_dump_json()) == profile


def test_legacy_profile_readable_but_update_requires_orbit():
    from uuid import uuid4

    body = payload()
    for field in ('apoapsis_km', 'periapsis_km', 'inclination_deg'):
        del body[field]
    stored = AdvisoryProfile.model_validate(body | dict(id=uuid4(), revision=1, updated_at=datetime.now(UTC)))
    assert stored.apoapsis_km is None
    with pytest.raises(ValidationError):
        AdvisorySpacecraftInput.model_validate(body)


@pytest.mark.parametrize('changes', [
    {'trajectory': {}}, {'reference': {}}, {'reference_mode': 'operator_reference'},
    {'policy': {'max_sample_gap_minutes': 15}},
    {'enabled': False},
])
def test_writes_reject_removed_inputs(changes):
    with pytest.raises(ValidationError):
        AdvisorySpacecraftInput.model_validate(payload() | changes)
