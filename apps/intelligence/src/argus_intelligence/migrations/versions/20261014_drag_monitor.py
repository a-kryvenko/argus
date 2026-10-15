"""Spacecraft profiles and immutable assessment snapshots."""
from alembic import op

revision = '20261014_drag_monitor'
down_revision = '20260914_intelligence'
branch_labels = None
depends_on = None


def upgrade():
    op.execute('''CREATE TABLE intelligence.drag_profile (
        id uuid PRIMARY KEY, owner_id bigint NOT NULL,
        revision integer NOT NULL DEFAULT 1 CHECK (revision > 0),
        payload jsonb NOT NULL, updated_at timestamptz NOT NULL DEFAULT now())''')
    op.execute('CREATE INDEX drag_profile_owner ON intelligence.drag_profile(owner_id)')
    op.execute('''CREATE TABLE intelligence.drag_run (
        id uuid PRIMARY KEY, profile_id uuid NOT NULL REFERENCES intelligence.drag_profile(id),
        profile_revision integer NOT NULL, release_id uuid,
        assessed_at timestamptz NOT NULL DEFAULT now(), payload jsonb NOT NULL)''')
    op.execute('CREATE INDEX drag_run_history ON intelligence.drag_run(profile_id, assessed_at DESC, id DESC)')


def downgrade():
    op.execute('DROP TABLE intelligence.drag_run')
    op.execute('DROP TABLE intelligence.drag_profile')
