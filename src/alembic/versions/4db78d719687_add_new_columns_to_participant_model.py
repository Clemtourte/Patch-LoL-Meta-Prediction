"""Add new columns to participant model

Revision ID: 4db78d719687
Revises: 4d76628d6011
Create Date: 2024-09-26 20:38:24.225536

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '4db78d719687'
down_revision = '4d76628d6011'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('participants', sa.Column('total_heal', sa.Integer(), nullable=True))
    op.add_column('participants', sa.Column('damage_taken', sa.Integer(), nullable=True))
    op.add_column('participants', sa.Column('wards_placed', sa.Integer(), nullable=True))
    op.add_column('participants', sa.Column('wards_killed', sa.Integer(), nullable=True))
    op.add_column('participants', sa.Column('time_ccing_others', sa.Integer(), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('participants', 'time_ccing_others')
    op.drop_column('participants', 'wards_killed')
    op.drop_column('participants', 'wards_placed')
    op.drop_column('participants', 'damage_taken')
    op.drop_column('participants', 'total_heal')
    # ### end Alembic commands ###