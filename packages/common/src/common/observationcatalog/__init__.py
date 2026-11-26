"""Metric definitions shared with the Clio and public APIs."""
from importlib.resources import files
import json


METRICS = {row['id']: row for row in json.loads(files(__package__).joinpath('catalog.json').read_text())['metrics']}
