import pandas as pd
import pytest
from clio.dataloaders.gfz_loader import parse_gfz_f107


def test_gfz_uses_observed_flux_and_rejects_missing_values():
    def row(day, observed):
        return ' '.join(map(str, [2026, 9, day, *([0]*22), observed, 999, 1]))
    frame = parse_gfz_f107('# GFZ daily observations\n' + row(1, 123.4) + '\n' + row(2, -1))
    assert frame.value.tolist() == [123.4]
    assert frame.observed_at.tolist() == [pd.Timestamp('2026-09-01T12:00Z')]
    with pytest.raises(ValueError, match='28 columns'):
        parse_gfz_f107('2026 09 01 1 2 3')

