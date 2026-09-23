import numpy as np
import pandas as pd
import pytest

from clio.dataloaders.omni_archive import WORDS, parse_annual


def record(year=2000, day=1, hour=0, fills=False):
    values = ['0'] * 55
    values[:3] = [str(year), str(day), str(hour)]
    fields = dict(bx=-5.6, by=2.2, bz=1.6, t=324194, n=2.9, v=675,
                  kp=53, dst=-45, ap=56, f10_7=125.6)
    if fills:
        fields.update(bx=999.9, by=999.9, bz=999.9, t=9999999, n=999.9,
                      v=9999, kp=99, dst=99999, ap=999, f10_7=999.9)
    for name, word in WORDS.items(): values[word-1] = str(fields[name])
    return ' '.join(values)


def test_word_mapping_kp_and_missing_values():
    d = parse_annual(record()+'\n'+record(hour=1, fills=True), 2000)
    assert d.issue_time.iloc[0] == pd.Timestamp('2000-01-01', tz='UTC')
    assert d.v.iloc[0] == 675
    assert d.by.iloc[0] == 2.2 and d.bz.iloc[0] == 1.6
    assert d.kp.iloc[0] == 5.3
    assert d.iloc[1].drop('issue_time').isna().all()


def test_leap_year_and_invalid_records():
    assert parse_annual(record(day=366,hour=23),2000).issue_time.iloc[0] == pd.Timestamp('2000-12-31 23:00',tz='UTC')
    for text in [record()+'\n'+record(), record(year=2001), record(hour=24), '2000 1 0']:
        with pytest.raises(ValueError): parse_annual(text,2000)
