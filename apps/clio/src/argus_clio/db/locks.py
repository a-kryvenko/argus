"""Distinct PostgreSQL advisory keys for source ingestion and scheduled jobs."""
SOURCE_LOCKS = {
    'solar_wind_mag': 730100,
    'solar_wind_plasma': 730101,
    'kp': 730110,
    'dst': 730111,
}
JOB_LOCKS = {'refresh': 730200, 'aggregate': 730201, 'aia': 730202}
