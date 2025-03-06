"""modify_spell_stats_table

Revision ID: d5e194f1991f
Revises: 5600b7cdf9c6
Create Date: 2025-01-30 16:36:53.511953

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic.
revision = 'd5e194f1991f'
down_revision = '5600b7cdf9c6'
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Modify spell_stats table
    with op.batch_alter_table('spell_stats') as batch_op:
        # Add new column first
        batch_op.add_column(sa.Column('effect_values', sa.JSON(), nullable=True))
        
        # Drop columns we don't need anymore
        batch_op.drop_column('ap_ratio')
        batch_op.drop_column('damage_values')
        batch_op.drop_column('resource')
        batch_op.drop_column('health_ratio')
        batch_op.drop_column('other_ratios')
        batch_op.drop_column('hp_ratio')
        batch_op.drop_column('cc_effects')
        batch_op.drop_column('ad_ratio')
        batch_op.drop_column('target_hp_ratio')
        batch_op.drop_column('mr_ratio')
        batch_op.drop_column('bonus_ad_ratio')
        batch_op.drop_column('armor_ratio')
        batch_op.drop_column('shield_amount')
        batch_op.drop_column('movement_speed_ratio')
        batch_op.drop_column('base_damage')
        batch_op.drop_column('cc_duration')
        batch_op.drop_column('heal_amount')

def downgrade() -> None:
    # Restore spell_stats columns
    with op.batch_alter_table('spell_stats') as batch_op:
        batch_op.add_column(sa.Column('heal_amount', sqlite.JSON(), nullable=True))
        batch_op.add_column(sa.Column('cc_duration', sqlite.JSON(), nullable=True))
        batch_op.add_column(sa.Column('base_damage', sqlite.JSON(), nullable=True))
        batch_op.add_column(sa.Column('movement_speed_ratio', sqlite.JSON(), nullable=True))
        batch_op.add_column(sa.Column('shield_amount', sqlite.JSON(), nullable=True))
        batch_op.add_column(sa.Column('armor_ratio', sqlite.JSON(), nullable=True))
        batch_op.add_column(sa.Column('bonus_ad_ratio', sqlite.JSON(), nullable=True))
        batch_op.add_column(sa.Column('mr_ratio', sqlite.JSON(), nullable=True))
        batch_op.add_column(sa.Column('target_hp_ratio', sqlite.JSON(), nullable=True))
        batch_op.add_column(sa.Column('ad_ratio', sqlite.JSON(), nullable=True))
        batch_op.add_column(sa.Column('cc_effects', sqlite.JSON(), nullable=True))
        batch_op.add_column(sa.Column('hp_ratio', sqlite.JSON(), nullable=True))
        batch_op.add_column(sa.Column('other_ratios', sqlite.JSON(), nullable=True))
        batch_op.add_column(sa.Column('health_ratio', sqlite.JSON(), nullable=True))
        batch_op.add_column(sa.Column('resource', sa.VARCHAR(), nullable=True))
        batch_op.add_column(sa.Column('damage_values', sqlite.JSON(), nullable=True))
        batch_op.add_column(sa.Column('ap_ratio', sqlite.JSON(), nullable=True))
        batch_op.drop_column('effect_values')