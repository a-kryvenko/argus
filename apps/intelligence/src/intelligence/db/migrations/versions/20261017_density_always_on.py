"""Resume all density profiles; every spacecraft is monitored."""
from alembic import op

revision = '20261017_density_always_on'
down_revision = '20261017_forecast_inputs'
branch_labels = None
depends_on = None


def upgrade():
    op.execute('''UPDATE intelligence.density_profile
        SET payload=jsonb_set(payload,'{enabled}','true'::jsonb),
            revision=revision+1, updated_at=now()
        WHERE payload->>'enabled' IS DISTINCT FROM 'true' ''')


def downgrade():
    pass
