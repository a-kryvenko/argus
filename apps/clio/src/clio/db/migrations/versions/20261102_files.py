"""Add generic files and nullable observation context to existing observation receipts."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = '20261102_files'
down_revision = '20261102_provider_contract'
branch_labels = None
depends_on = None
# S3-only receipts cannot be read by the previous local-archive application.
online_compatible = False


def success_constraint():
    if not op.get_context().as_sql:
        constraints = sa.inspect(op.get_bind()).get_check_constraints('file_observation', schema='clio')
        return next(item['name'] for item in constraints if item['name'].endswith('file_observation_success_file'))
    return 'ck_file_observation_file_observation_success_file'


def upgrade():
    op.create_table('files',
        sa.Column('id', sa.Uuid(), primary_key=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('bucket', sa.String(63), nullable=False),
        sa.Column('key', sa.Text(), nullable=False),
        sa.UniqueConstraint('bucket', 'key', name='uq_files_bucket_key'), schema='clio')
    op.create_index('ix_files_timestamp', 'files', ['timestamp'], schema='clio')
    for column in (sa.Column('file_id', sa.Uuid()), sa.Column('variable', sa.String(32)),
                   sa.Column('observed_at', sa.DateTime(timezone=True)), sa.Column('entity', sa.String(64)),
                   sa.Column('details', JSONB())):
        op.add_column('file_observation', column, schema='clio')
    op.create_foreign_key('fk_file_observation_file_id_files', 'file_observation', 'files', ['file_id'], ['id'],
                          source_schema='clio', referent_schema='clio')
    name = success_constraint()
    op.drop_constraint(op.f(name), 'file_observation', type_='check', schema='clio')
    op.create_check_constraint(op.f(name), 'file_observation',
        'error IS NOT NULL OR raw_path IS NOT NULL OR file_id IS NOT NULL', schema='clio')
    op.create_check_constraint('file_observation_context', 'file_observation',
        'file_id IS NULL OR entity IS NOT NULL', schema='clio')
    op.create_unique_constraint('uq_file_observation_file_context', 'file_observation',
                               ['file_id', 'variable', 'observed_at', 'entity'], schema='clio')
    op.create_index('ix_file_observation_variable_observed', 'file_observation', ['variable', 'observed_at'], schema='clio')


def downgrade():
    raise RuntimeError('Keep file references while a deployed release may use them')
