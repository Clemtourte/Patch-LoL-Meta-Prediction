"""add previous columns to spell stats

Revision ID: 5ffc9c4145f5
Revises: b2ceb2aa5165
Create Date: 2025-03-07 19:45:57.845225

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic.
revision = '5ffc9c4145f5'
down_revision = 'b2ceb2aa5165'
branch_labels = None
depends_on = None


def upgrade():
    # Add new columns for previous values
    with op.batch_alter_table('spell_stats') as batch_op:
        batch_op.add_column(sa.Column('previous_base_damage', sqlite.JSON(), nullable=True))
        batch_op.add_column(sa.Column('previous_ap_ratio', sqlite.JSON(), nullable=True))
        batch_op.add_column(sa.Column('previous_ad_ratio', sqlite.JSON(), nullable=True))
        batch_op.add_column(sa.Column('previous_bonus_ad_ratio', sqlite.JSON(), nullable=True))
        batch_op.add_column(sa.Column('previous_max_health_ratio', sqlite.JSON(), nullable=True))
        batch_op.add_column(sa.Column('previous_cooldown', sqlite.JSON(), nullable=True))
        batch_op.add_column(sa.Column('previous_mana_cost', sqlite.JSON(), nullable=True))
        batch_op.add_column(sa.Column('previous_range', sqlite.JSON(), nullable=True))
        batch_op.add_column(sa.Column('previous_shield_value', sqlite.JSON(), nullable=True))
        batch_op.add_column(sa.Column('previous_shield_ratio', sqlite.JSON(), nullable=True))
        batch_op.add_column(sa.Column('previous_heal_value', sqlite.JSON(), nullable=True))
        batch_op.add_column(sa.Column('previous_heal_ratio', sqlite.JSON(), nullable=True))
        batch_op.add_column(sa.Column('previous_slow_percent', sqlite.JSON(), nullable=True))
        batch_op.add_column(sa.Column('previous_stun_duration', sqlite.JSON(), nullable=True))
        batch_op.add_column(sa.Column('previous_knockup_duration', sqlite.JSON(), nullable=True))
        batch_op.add_column(sa.Column('previous_root_duration', sqlite.JSON(), nullable=True))
        batch_op.add_column(sa.Column('previous_damage_type', sa.String(), nullable=True))
        batch_op.add_column(sa.Column('previous_resource_type', sa.String(), nullable=True))

    # Create a data migration function to populate new columns from previous_values
    op.execute("""
    -- This is a placeholder for a data migration if needed.
    -- We could write a function to populate the new columns from the previous_values JSON
    -- But this might be better handled in Python code after the migration
    """)


def downgrade():
    # Remove the added columns
    with op.batch_alter_table('spell_stats') as batch_op:
        batch_op.drop_column('previous_base_damage')
        batch_op.drop_column('previous_ap_ratio')
        batch_op.drop_column('previous_ad_ratio')
        batch_op.drop_column('previous_bonus_ad_ratio')
        batch_op.drop_column('previous_max_health_ratio')
        batch_op.drop_column('previous_cooldown')
        batch_op.drop_column('previous_mana_cost')
        batch_op.drop_column('previous_range')
        batch_op.drop_column('previous_shield_value')
        batch_op.drop_column('previous_shield_ratio')
        batch_op.drop_column('previous_heal_value')
        batch_op.drop_column('previous_heal_ratio')
        batch_op.drop_column('previous_slow_percent')
        batch_op.drop_column('previous_stun_duration')
        batch_op.drop_column('previous_knockup_duration')
        batch_op.drop_column('previous_root_duration')
        batch_op.drop_column('previous_damage_type')
        batch_op.drop_column('previous_resource_type')