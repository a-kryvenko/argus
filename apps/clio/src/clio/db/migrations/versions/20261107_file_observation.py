"""Add the original-file observation contract without removing legacy receipts."""
from alembic import op
import sqlalchemy as sa

revision = '20261107_file_observation'
down_revision = '20261102_files'
branch_labels = None
depends_on = None
# Legacy fields remain available, and both receipt formats are accepted.
online_compatible = True


def upgrade():
    op.add_column('file_observation', sa.Column('observation', sa.String(64)), schema='clio')
    for column, type_ in (('kind', sa.String(32)), ('source_key', sa.Text()),
                          ('received_at', sa.DateTime(timezone=True))):
        op.alter_column('file_observation', column, existing_type=type_, nullable=True, schema='clio')

    # New receipts have a checksum and file_id, but no local raw_path or entity.
    constraints = (sa.inspect(op.get_bind()).get_check_constraints('file_observation', schema='clio')
                   if not op.get_context().as_sql else [])
    for suffix, condition in (
        ('file_observation_file_checksum',
         'observation IS NOT NULL OR (raw_path IS NULL) = (sha256 IS NULL)'),
        ('file_observation_context',
         'observation IS NOT NULL OR file_id IS NULL OR entity IS NOT NULL'),
    ):
        name = next((item['name'] for item in constraints if item['name'].endswith(suffix)),
                    'ck_file_observation_' + suffix)
        op.drop_constraint(op.f(name), 'file_observation', type_='check', schema='clio')
        op.create_check_constraint(op.f(name), 'file_observation', condition, schema='clio')

    # Only completed originals can satisfy the new API. Keep derivatives, archive
    # receipts and duplicate polls unchanged rather than deleting their history.
    op.execute("""WITH originals AS (
        SELECT id, file_id, received_at,
            CASE
                WHEN kind = 'sdo' AND variable ~ '^aia[0-9]+$'
                    THEN 'aia_' || lpad(substring(variable FROM '[0-9]+$'), 3, '0') || '_nrt'
                WHEN kind = 'sdo' AND variable = 'hmi_m' THEN 'hmi_m_nrt'
                WHEN kind = 'sdo_browse' AND variable ~ '^aia[0-9]+$'
                    AND details->>'complete' = 'true'
                    THEN 'aia_' || lpad(substring(variable FROM '[0-9]+$'), 3, '0') || '_browse'
                WHEN kind = 'donki' THEN 'donki.' || source_product
                WHEN kind IN ('gong', 'goes', 'enlil') THEN kind
            END AS observation,
            COALESCE(sha256, details->>'sha256') AS checksum,
            COALESCE(observed_at, received_at) AS observed
        FROM clio.file_observation
        WHERE file_id IS NOT NULL AND error IS NULL AND entity LIKE '%.original'
    ), ranked AS (
        SELECT *, row_number() OVER (PARTITION BY observation, file_id ORDER BY received_at DESC, id DESC) AS rank
        FROM originals
        WHERE observation IS NOT NULL AND checksum ~ '^[0-9a-f]{64}$' AND observed IS NOT NULL
    )
    UPDATE clio.file_observation AS receipt
    SET observation = original.observation, sha256 = original.checksum, observed_at = original.observed
    FROM ranked AS original WHERE receipt.id = original.id AND original.rank = 1""")

    op.create_check_constraint('file_observation_original_fields', 'file_observation',
        "observation IS NULL OR (file_id IS NOT NULL AND observed_at IS NOT NULL "
        "AND sha256 IS NOT NULL AND sha256 ~ '^[0-9a-f]{64}$')", schema='clio')
    op.create_unique_constraint('uq_file_observation_original', 'file_observation',
                               ['observation', 'file_id'], schema='clio')
    op.create_index('ix_file_observation_observation_observed', 'file_observation',
                    ['observation', 'observed_at'], schema='clio')


def downgrade():
    raise RuntimeError('Keep original-file observations while a deployed release may use them')
