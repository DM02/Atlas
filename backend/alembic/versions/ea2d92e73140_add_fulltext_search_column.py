"""add fulltext search column

Revision ID: ea2d92e73140
Revises: 5d0b44177321
Create Date: 2026-08-11 21:01:13.498106

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ea2d92e73140'
down_revision: Union[str, Sequence[str], None] = '5d0b44177321'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute(
        "ALTER TABLE document_chunks "
        "ADD COLUMN content_tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', content)) STORED"
    )
    op.execute(
        "CREATE INDEX document_chunks_content_tsv_idx ON document_chunks USING gin (content_tsv)"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX IF EXISTS document_chunks_content_tsv_idx")
    op.execute("ALTER TABLE document_chunks DROP COLUMN IF EXISTS content_tsv")
