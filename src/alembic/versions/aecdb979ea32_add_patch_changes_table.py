"""add_patch_changes_table

Revision ID: aecdb979ea32
Revises: 0cd7c9a95067
Create Date: 2025-01-23 23:34:14.488337

"""
from alembic import op
import sqlalchemy as sa

# Add these two lines:
revision = 'aecdb979ea32'
down_revision = '0cd7c9a95067'  # The previous revision ID

branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'patch_changes',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('from_patch', sa.String, nullable=False),
        sa.Column('to_patch', sa.String, nullable=False),
        sa.Column('champion_name', sa.String, nullable=False),
        sa.Column('stat_type', sa.String, nullable=False),
        sa.Column('stat_name', sa.String, nullable=False),
        sa.Column('change_value', sa.Float, nullable=False)
    )
    
    op.create_index(
        'patch_champion_idx',
        'patch_changes',
        ['from_patch', 'to_patch', 'champion_name']
    )

def downgrade():
    op.drop_index('patch_champion_idx')
    op.drop_table('patch_changes')