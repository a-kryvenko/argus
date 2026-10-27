"""Shared selection of the latest published release for each product."""

LATEST_RELEASES = '''SELECT DISTINCT ON (r.product) r.*
    FROM forecast_release r JOIN forecast_run f ON f.id=r.run_id
    ORDER BY r.product,r.issue_time DESC,f.started_at DESC,r.published_at DESC,r.id DESC'''
