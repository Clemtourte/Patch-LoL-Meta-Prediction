"""add perf score columns

Revision ID: 07eb7628af35
Revises: 4db78d719687
Create Date: 2024-10-02 23:02:37.374846

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '07eb7628af35'
down_revision = '4db78d719687'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('participants', sa.Column('performance_score', sa.Float(), nullable=True))
    op.add_column('participants', sa.Column('standardized_performance_score', sa.Float(), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('participants', 'standardized_performance_score')
    op.drop_column('participants', 'performance_score')
    # ### end Alembic commands ###