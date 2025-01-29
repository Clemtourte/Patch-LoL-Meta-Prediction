"""add_spell_stats_columns

Revision ID: da4b90f63861
Revises: 9a0071ba6314
Create Date: 2025-01-28 16:25:33.116540

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = 'da4b90f63861'
down_revision = '9a0071ba6314'
branch_labels = None
depends_on = None


def upgrade():
    # First, create new columns
    op.add_column('spell_stats', sa.Column('base_damage', sa.JSON))
    op.add_column('spell_stats', sa.Column('ad_ratio', sa.JSON))
    op.add_column('spell_stats', sa.Column('ap_ratio', sa.JSON))
    op.add_column('spell_stats', sa.Column('cc_duration', sa.JSON))
    op.add_column('spell_stats', sa.Column('heal_amount', sa.JSON))
    op.add_column('spell_stats', sa.Column('shield_amount', sa.JSON))
    op.add_column('spell_stats', sa.Column('movement_speed_ratio', sa.JSON))
    op.add_column('spell_stats', sa.Column('health_ratio', sa.JSON))
    op.add_column('spell_stats', sa.Column('other_ratios', sa.JSON))
    
    # Create a new standardized_id column
    op.add_column('spell_stats', sa.Column('standardized_id', sa.String(100)))
    
    # Create index on standardized_id
    op.create_index('idx_spell_stats_standardized_id', 'spell_stats', ['standardized_id'])

def downgrade():
    # Remove added columns in reverse order
    op.drop_index('idx_spell_stats_standardized_id')
    op.drop_column('spell_stats', 'standardized_id')
    op.drop_column('spell_stats', 'other_ratios')
    op.drop_column('spell_stats', 'health_ratio')
    op.drop_column('spell_stats', 'movement_speed_ratio')
    op.drop_column('spell_stats', 'shield_amount')
    op.drop_column('spell_stats', 'heal_amount')
    op.drop_column('spell_stats', 'cc_duration')
    op.drop_column('spell_stats', 'ap_ratio')
    op.drop_column('spell_stats', 'ad_ratio')
    op.drop_column('spell_stats', 'base_damage')