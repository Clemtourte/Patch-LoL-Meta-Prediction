"""add_more_spell_stats_columns

Revision ID: 5600b7cdf9c6
Revises: da4b90f63861
Create Date: 2025-01-29 01:47:56.155075

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '5600b7cdf9c6'
down_revision = 'da4b90f63861'
branch_labels = None
depends_on = None


def upgrade():
    # Add new columns
    with op.batch_alter_table('spell_stats') as batch_op:
        batch_op.add_column(sa.Column('bonus_ad_ratio', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('hp_ratio', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('target_hp_ratio', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('armor_ratio', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('mr_ratio', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('resource_type', sa.String(), nullable=True))
        batch_op.add_column(sa.Column('cc_effects', sa.JSON(), nullable=True))

def downgrade():
    # Remove the columns if needed
    with op.batch_alter_table('spell_stats') as batch_op:
        batch_op.drop_column('bonus_ad_ratio')
        batch_op.drop_column('hp_ratio')
        batch_op.drop_column('target_hp_ratio')
        batch_op.drop_column('armor_ratio')
        batch_op.drop_column('mr_ratio')
        batch_op.drop_column('resource_type')
        batch_op.drop_column('cc_effects')