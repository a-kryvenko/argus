import pytest


@pytest.mark.parametrize('domain', ['api', 'clio', 'prophet', 'intelligence'])
def test_raw_passwords_and_domain_settings_are_isolated(domain, monkeypatch):
    from sqlalchemy.engine import make_url
    from common.db.session import get_database_url
    get_url = lambda: get_database_url(domain.upper())
    prefix = domain.upper() + '_DB_'
    for field in ('NAME', 'USER', 'PASSWORD', 'HOST', 'PORT'):
        monkeypatch.delenv(prefix + field, raising=False)
    monkeypatch.setenv('DB_USER', 'postgres')
    monkeypatch.setenv('DB_PASSWORD', 'admin-secret')
    monkeypatch.setenv('DATABASE_URL', 'postgresql://admin:secret@db/admin')
    with pytest.raises(RuntimeError, match=prefix + 'PASSWORD'):
        get_url()
    for field, value in dict(NAME=domain, USER='owner', PASSWORD='temporary', PORT='5433').items():
        monkeypatch.setenv(prefix + field, value)
    for password in ('p@ss/word', 'a:b#c?d%20+e', 'space and $dollar', 'пароль=тест'):
        monkeypatch.setenv(prefix + 'PASSWORD', password)
        url = get_url()
        assert url.database == domain and url.username == 'owner'
        assert url.host == 'localhost' and url.port == 5433
        assert url.password == password
        assert make_url(url.render_as_string(hide_password=False)).password == password
    for invalid in ('notaport', '0', '65536'):
        monkeypatch.setenv(prefix + 'PORT', invalid)
        with pytest.raises(RuntimeError, match=prefix + 'PORT') as exc:
            get_url()
        assert password not in str(exc.value)
