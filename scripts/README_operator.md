# Environment Setup (UV)

This project uses **uv** for dependency management.

## Install uv

```bash
pip install uv
```

# Operator commands

Normal flow:

```bash
uv venv
source .venv/bin/activate
uv sync

uv run python scripts/manage.py setup
uv run python scripts/manage.py prepare-forecast
uv run python scripts/manage.py serve-public
```

* For SDO data sync you may need to register your email

http://jsoc.stanford.edu/ajax/register_email.html


Smoke request while API is running:

```bash
uv run python scripts/manage.py forecast-smoke
```

All operator-facing paths are controlled by `configs/project.yaml`.
Low-level scripts remain internal implementation details.
