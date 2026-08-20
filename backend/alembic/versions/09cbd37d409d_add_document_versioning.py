"""add document versioning

Revision ID: 09cbd37d409d
Revises: 58ce56256af0
Create Date: 2026-08-12 13:08:00.470354

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '09cbd37d409d'
down_revision: Union[str, Sequence[str], None] = '58ce56256af0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # NOTE: autogenerate wants to drop+recreate document_chunks_content_tsv_idx (gin) and
    # document_chunks_embedding_hnsw_idx (hnsw) here — same false positive as prior
    # migrations (it doesn't recognize index types created via raw op.execute()).
    # Deliberately not touching them.
    op.add_column('documents', sa.Column('current_version_id', sa.Uuid(), nullable=True))
    op.create_foreign_key(None, 'documents', 'document_versions', ['current_version_id'], ['id'], ondelete='SET NULL')
    op.add_column('ingestion_jobs', sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False))

    # Backfill: existing documents were ingested synchronously (Phase 1-3), always
    # exactly one version, current_version_id was never a thing before this migration.
    op.execute(
        "UPDATE documents SET current_version_id = document_versions.id "
        "FROM document_versions "
        "WHERE document_versions.document_id = documents.id "
        "AND documents.status = 'ready'"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('ingestion_jobs', 'created_at')
    op.drop_constraint(None, 'documents', type_='foreignkey')
    op.drop_column('documents', 'current_version_id')
    # ### end Alembic commands ###
