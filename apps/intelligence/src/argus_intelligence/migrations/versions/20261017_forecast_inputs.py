"""Retain the exact published inputs of density assessments in the owning DB."""
from alembic import op

revision = '20261017_forecast_inputs'
down_revision = '20261016_density_advisory'
branch_labels = None
depends_on = None


def upgrade():
    op.execute('''CREATE TABLE intelligence.forecast_input (
        release_id uuid PRIMARY KEY, product text NOT NULL,
        sha256 text NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
        content_gzip bytea NOT NULL, captured_at timestamptz NOT NULL,
        UNIQUE (release_id, product))''')
    op.execute('''CREATE TABLE intelligence.density_run_input (
        run_id uuid NOT NULL REFERENCES intelligence.density_run(id),
        product text NOT NULL, release_id uuid NOT NULL,
        PRIMARY KEY (run_id, product),
        FOREIGN KEY (release_id, product)
            REFERENCES intelligence.forecast_input(release_id, product))''')


def downgrade():
    op.execute('DROP TABLE intelligence.density_run_input')
    op.execute('DROP TABLE intelligence.forecast_input')
