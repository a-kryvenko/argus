"""Regression coverage for PostgreSQL constraint deparsing during transfer."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'transfer_checks', Path(__file__).resolve().parents[1] / 'scripts/transfer-databases.py')
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)


def normalized(definition):
    return transfer.normalize_snapshot({'constraints': [['forecast_run', 'run_status', definition]]})


@pytest.mark.parametrize('column,values', [
    ('status', ['stored', 'skipped']),
    ('status', ['running', 'succeeded', 'partial', 'failed', 'interrupted']),
    ('trigger', ['manual', 'scheduled']),
    ('status', ['running', 'succeeded', 'partial', 'failed', 'interrupted', 'imported']),
])
def test_dump_restore_enum_cast_forms_match(column, values):
    elements = [f"'{value}'::character varying" for value in values]
    source = f"CHECK ((({column})::text = ANY ((ARRAY[{', '.join(elements)}])::text[])))"
    casts = ', '.join(f'({element})::text' for element in elements)
    restored = f'CHECK ((({column})::text = ANY (ARRAY[{casts}])))'
    assert normalized(source) == normalized(restored)
    assert normalized(restored)['constraints'][0][2] == restored
    assert normalized(source) != normalized(restored.replace(values[0], 'different_value'))
    assert normalized(source) != normalized(restored.replace('= ANY', '<> ALL'))


@pytest.mark.parametrize('definition', [
    "CHECK (attempts > 0)",
    "CHECK (((status)::text = ANY ((ARRAY['a,b'::character varying])::text[])))",
    "CHECK (((status)::text = ANY ((ARRAY['a'::character varying(1)])::text[])))",
    "CHECK (((status)::text = ANY ((ARRAY['a'::character varying])::integer[])))",
    "CHECK (((status)::text = ANY ((ARRAY['a'::character varying])::text[]))) NOT VALID",
])
def test_other_definitions_are_preserved(definition):
    assert normalized(definition)['constraints'][0][2] == definition
