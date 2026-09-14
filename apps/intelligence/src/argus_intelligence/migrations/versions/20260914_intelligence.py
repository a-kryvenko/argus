"""Intelligence attempts and completed release results, owned by this domain."""
from alembic import op

revision = '20260914_intelligence'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""CREATE TABLE intelligence.attempt (
        id uuid PRIMARY KEY, product text NOT NULL, release_id uuid, prophet_run_id uuid,
        status text NOT NULL CHECK(status IN ('running','succeeded','skipped','failed','interrupted')),
        started_at timestamptz NOT NULL DEFAULT now(), finished_at timestamptz, error text)""")
    op.execute('CREATE INDEX attempt_product_started ON intelligence.attempt(product,started_at DESC)')
    op.execute("""CREATE TABLE intelligence.result (
        product text NOT NULL, release_id uuid NOT NULL, prophet_run_id uuid NOT NULL,
        attempt_id uuid NOT NULL REFERENCES intelligence.attempt(id),
        processed_at timestamptz NOT NULL DEFAULT now(), result jsonb NOT NULL,
        PRIMARY KEY(product,release_id))""")


def downgrade():
    op.execute('DROP TABLE intelligence.result')
    op.execute('DROP TABLE intelligence.attempt')
