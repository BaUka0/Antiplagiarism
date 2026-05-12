from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '5dc5fe7e8c51'
down_revision: Union[str, Sequence[str], None] = '46787cece54e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_unique_constraint('uq_document_filename_year', 'documents', ['filename', 'year'])


def downgrade() -> None:
    op.drop_constraint('uq_document_filename_year', 'documents', type_='unique')
