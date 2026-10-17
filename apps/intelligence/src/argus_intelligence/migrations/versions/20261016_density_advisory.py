"""Supplied-trajectory density advisory profiles, snapshots and journal."""
from alembic import op

revision='20261016_density_advisory'
down_revision='20261015_drag_events'
branch_labels=None
depends_on=None

online_compatible = True


def upgrade():
    op.execute('''CREATE TABLE intelligence.density_profile (
        id uuid PRIMARY KEY,owner_id bigint NOT NULL,revision integer NOT NULL DEFAULT 1 CHECK(revision>0),
        payload jsonb NOT NULL,updated_at timestamptz NOT NULL DEFAULT now())''')
    op.execute('CREATE INDEX density_profile_owner ON intelligence.density_profile(owner_id)')
    op.execute('''CREATE TABLE intelligence.density_run (
        id uuid PRIMARY KEY,profile_id uuid NOT NULL REFERENCES intelligence.density_profile(id),
        profile_revision integer NOT NULL,assessed_at timestamptz NOT NULL,payload jsonb NOT NULL)''')
    op.execute('CREATE INDEX density_run_history ON intelligence.density_run(profile_id,assessed_at DESC,id DESC)')
    op.execute('''CREATE TABLE intelligence.density_event (
        id uuid PRIMARY KEY,profile_id uuid NOT NULL REFERENCES intelligence.density_profile(id),
        run_id uuid NOT NULL REFERENCES intelligence.density_run(id),detected_at timestamptz NOT NULL,
        payload jsonb NOT NULL,acknowledged_at timestamptz)''')
    op.execute('CREATE INDEX density_event_history ON intelligence.density_event(profile_id,detected_at DESC,id DESC)')
    op.execute('''CREATE TABLE intelligence.density_event_state (
        profile_id uuid PRIMARY KEY REFERENCES intelligence.density_profile(id),payload jsonb NOT NULL)''')


def downgrade():
    for table in ('density_event_state','density_event','density_run','density_profile'):
        op.execute(f'DROP TABLE intelligence.{table}')
