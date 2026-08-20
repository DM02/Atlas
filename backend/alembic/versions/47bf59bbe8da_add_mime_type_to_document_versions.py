"""add mime_type to document_versions

Revision ID: 47bf59bbe8da
Revises: 09cbd37d409d
Create Date: 2026-08-12 13:24:29.169512

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '47bf59bbe8da'
down_revision: Union[str, Sequence[str], None] = '09cbd37d409d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # NOTE: autogenerate wants to drop+recreate the two document_chunks indexes here —
    # same recurring false positive (raw op.execute()-created indexes aren't recognized).
    # Deliberately not touching them.
    op.add_column('document_versions', sa.Column('mime_type', sa.String(length=128), nullable=True))

    # Backfill: mime_type used to live only on documents (one version each, pre-Phase-4).
    op.execute(
        "UPDATE document_versions SET mime_type = documents.mime_type "
        "FROM documents WHERE documents.id = document_versions.document_id"
    )

    op.alter_column('document_versions', 'mime_type', nullable=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('document_versions', 'mime_type')
