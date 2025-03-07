"""add change_text column to patch_changes

Revision ID: b2ceb2aa5165
Revises: a7158697f812
Create Date: 2025-03-07 18:10:23.870379

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic.
revision = 'b2ceb2aa5165'
down_revision = 'a7158697f812'
branch_labels = None
depends_on = None


from alembic import op
import sqlalchemy as sa

def upgrade():
    # Create temporary table for any existing data
    op.execute('CREATE TABLE spell_stats_backup AS SELECT * FROM spell_stats')
    
    # Drop current table
    op.drop_table('spell_stats')
    
    # Create the new table
    op.create_table('spell_stats',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('version', sa.String(), nullable=False),
        sa.Column('champion', sa.String(), nullable=False),
        sa.Column('spell_type', sa.String(), nullable=False),
        sa.Column('spell_name', sa.String(), nullable=False),
        sa.Column('base_damage', sqlite.JSON(), nullable=True),
        sa.Column('ap_ratio', sqlite.JSON(), nullable=True),
        sa.Column('ad_ratio', sqlite.JSON(), nullable=True),
        sa.Column('bonus_ad_ratio', sqlite.JSON(), nullable=True),
        sa.Column('max_health_ratio', sqlite.JSON(), nullable=True),
        sa.Column('cooldown', sqlite.JSON(), nullable=True),
        sa.Column('mana_cost', sqlite.JSON(), nullable=True),
        sa.Column('range', sqlite.JSON(), nullable=True),
        sa.Column('shield_value', sqlite.JSON(), nullable=True),
        sa.Column('shield_ratio', sqlite.JSON(), nullable=True),
        sa.Column('heal_value', sqlite.JSON(), nullable=True),
        sa.Column('heal_ratio', sqlite.JSON(), nullable=True),
        sa.Column('slow_percent', sqlite.JSON(), nullable=True),
        sa.Column('stun_duration', sqlite.JSON(), nullable=True),
        sa.Column('knockup_duration', sqlite.JSON(), nullable=True),
        sa.Column('root_duration', sqlite.JSON(), nullable=True),
        sa.Column('damage_type', sa.String(), nullable=True),
        sa.Column('resource_type', sa.String(), nullable=True),
        sa.Column('description', sa.String(), nullable=True),
        sa.Column('last_changed', sa.String(), nullable=True),
        sa.Column('previous_values', sqlite.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create index
    op.create_index('idx_spell_stats_champ_spell', 'spell_stats', 
                   ['champion', 'spell_type', 'version'], unique=False)

def downgrade():
    # Drop the new table
    op.drop_table('spell_stats')
    
    # Restore from backup
    op.execute('CREATE TABLE spell_stats AS SELECT * FROM spell_stats_backup')
    op.execute('DROP TABLE spell_stats_backup')
    
    # Recreate the original index
    op.create_index('idx_spell_stats_champ_spell', 'spell_stats', 
                   ['champion', 'spell_type', 'version'], unique=False)