"""Durable event journal and opaque private-core transition state."""
from alembic import op

revision = '20261015_drag_events'
down_revision = '20261014_drag_monitor'
branch_labels = None
depends_on = None


def upgrade():
    op.execute('''CREATE TABLE intelligence.drag_event_state (
        profile_id uuid PRIMARY KEY REFERENCES intelligence.drag_profile(id),
        payload jsonb NOT NULL, updated_at timestamptz NOT NULL DEFAULT now())''')
    op.execute('''CREATE TABLE intelligence.drag_event (
        id uuid PRIMARY KEY, profile_id uuid NOT NULL REFERENCES intelligence.drag_profile(id),
        run_id uuid NOT NULL REFERENCES intelligence.drag_run(id),
        detected_at timestamptz NOT NULL, payload jsonb NOT NULL,
        acknowledged_at timestamptz, acknowledged_by bigint)''')
    op.execute('CREATE INDEX drag_event_profile_time ON intelligence.drag_event(profile_id,detected_at DESC,id DESC)')
    op.execute('CREATE INDEX drag_event_time ON intelligence.drag_event(detected_at DESC,id DESC)')


def downgrade():
    op.execute('DROP TABLE intelligence.drag_event')
    op.execute('DROP TABLE intelligence.drag_event_state')
