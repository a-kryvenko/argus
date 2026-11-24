"""PostgreSQL catalogs read saved references without downloading objects."""
from datetime import UTC, datetime, timedelta

import pytest


@pytest.mark.parametrize('kind', ['gong', 'goes'])
def test_snapshot_catalog_reads_metadata_without_object_downloads(file_api, tmp_path, kind):
    from clio.ingestion.utils.file_observations import save
    observed_at = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    path = tmp_path / 'original.fits'
    path.write_bytes(b'original')
    save(path, observation=kind, observed_at=observed_at, source_product=kind+'.live')

    def catalog(start):
        response = file_api.get('/internal/v1/files', params={
            'observation': kind, 'start': start.isoformat(), 'end': (start+timedelta(hours=1)).isoformat(),
        })
        assert response.status_code == 200, response.text
        return response.json()['data']

    refs = catalog(observed_at)
    assert len(refs) == 1 and refs[0]['file']['bucket'] == 'argus-private'
    assert catalog(observed_at+timedelta(hours=1)) == []
