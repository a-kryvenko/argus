import hashlib
import json
from uuid import uuid4

import httpx
import pytest
from argus_intelligence import cli


@pytest.fixture
def payload(monkeypatch):
    monkeypatch.setenv('FORECASTS_URL', 'http://prophet-api:8000')
    monkeypatch.setenv('FORECASTS_SERVICE_TOKEN', 'secret-token')
    csv = 'issue_time,valid_time,lead_hours,dst_q10,dst_q50,dst_q90\n2026-09-13T00:00:00Z,2026-09-13T01:00:00Z,1,-20,-10,0\n'
    return {'contract_version': 1, 'product': 'dst', 'release_id': str(uuid4()), 'run_id': str(uuid4()),
            'published_at': '2026-09-13T00:00:00Z', 'issue_time': '2026-09-13T00:00:00Z',
            'artifacts': [{'name': 'dst_quantile', 'csv_text': csv, 'sha256': hashlib.sha256(csv.encode()).hexdigest(),
                           'row_count': 1, 'columns': csv.splitlines()[0].split(','), 'model_info': {}}]}


@pytest.mark.parametrize('pinned', [False, True])
def test_valid_release_is_identified_without_claiming_risk_readiness(payload, pinned):
    def respond(request):
        assert request.headers['Authorization'] == 'Bearer secret-token'
        suffix = f"releases/{payload['release_id']}" if pinned else 'latest'
        assert request.url.path == '/internal/v1/forecasts/dst/' + suffix
        return httpx.Response(200, json=payload)
    result = cli.check('dst', release_id=payload['release_id'] if pinned else None,
                       transport=httpx.MockTransport(respond))
    assert result['release_id'] == payload['release_id']
    assert result['run_id'] == payload['run_id']
    assert result['status'] == 'ok' and result['mode'] == 'stub'
    assert result['risk_assessment'] is None


@pytest.mark.parametrize('failure', ['checksum', 'version', 'missing', 'wrong-release', 'wrong-product', 'too-large', 'unauthorized', 'unavailable'])
def test_invalid_inputs_never_report_success(payload, failure, monkeypatch):
    product, release_id = 'dst', None
    if failure == 'checksum': payload['artifacts'][0]['csv_text'] += 'bad'
    if failure == 'version': payload['contract_version'] = 2
    if failure == 'missing': payload['artifacts'] = []
    if failure == 'wrong-release': release_id = uuid4()
    if failure == 'wrong-product': product = 'hmf'
    if failure == 'too-large': monkeypatch.setattr(cli, 'MAX_RESPONSE_BYTES', 10)
    status = 401 if failure == 'unauthorized' else 503 if failure == 'unavailable' else 200
    with pytest.raises((ValueError, httpx.HTTPError)):
        cli.check(product, release_id=release_id, transport=httpx.MockTransport(lambda r: httpx.Response(status, json=payload)))


def test_cli_failure_is_nonzero_and_does_not_leak_inputs(monkeypatch, capsys):
    monkeypatch.setattr('sys.argv', ['intelligence', 'check'])
    def fail(*args, **kwargs): raise ValueError('secret-token and payload')
    monkeypatch.setattr(cli, 'check', fail)
    with pytest.raises(SystemExit) as result:
        cli.main()
    assert result.value.code == 1
    output = capsys.readouterr().out
    assert 'secret-token' not in output and json.loads(output)['status'] == 'error'
