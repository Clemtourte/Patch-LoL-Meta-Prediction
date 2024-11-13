"""add champion_role_patch column

Revision ID: 3894b4094122
Revises: 865f3db2f027
Create Date: 2024-01-17 10:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '3894b4094122'
down_revision: Union[str, None] = '865f3db2f027'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # Simply add the column
    op.add_column('performance_features',
        sa.Column('champion_role_patch', sa.String(), nullable=True)
    )

def downgrade() -> None:
    # Drop the column if needed
    op.drop_column('performance_features', 'champion_role_patch')