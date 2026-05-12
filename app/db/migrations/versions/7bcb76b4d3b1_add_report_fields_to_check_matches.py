from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "7bcb76b4d3b1"
down_revision: Union[str, Sequence[str], None] = "5dc5fe7e8c51"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("check_matches", sa.Column("corpus_doc_idx", sa.Integer(), nullable=False))
    op.add_column(
        "check_match_chunks",
        sa.Column("corpus_chunk_global", sa.Integer(), nullable=False),
    )


def downgrade() -> None:
    op.drop_column("check_match_chunks", "corpus_chunk_global")
    op.drop_column("check_matches", "corpus_doc_idx")
