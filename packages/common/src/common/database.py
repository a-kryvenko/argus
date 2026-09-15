"""Read raw PostgreSQL settings without URL parsing or secret interpolation."""
import os


def database_parameters(prefix: str) -> dict:
    """An empty prefix is reserved for the administrative provisioning tool."""
    base = f'{prefix}_DB_' if prefix else 'DB_'
    required = [base + field for field in ('NAME', 'USER', 'PASSWORD')]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise ValueError('Missing database settings: ' + ', '.join(missing))
    host = os.getenv(base + 'HOST', 'localhost')
    if not host:
        raise ValueError(base + 'HOST must not be empty')
    try:
        port = int(os.getenv(base + 'PORT', '5432'))
        if not 1 <= port <= 65535:
            raise ValueError()
    except ValueError:
        raise ValueError(base + 'PORT must be an integer between 1 and 65535') from None
    return dict(host=host, port=port, database=os.environ[base + 'NAME'],
                username=os.environ[base + 'USER'], password=os.environ[base + 'PASSWORD'])
