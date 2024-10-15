"""Update spell_stats table

Revision ID: 582c0f14a951
Revises: 44c14d50b0ef
Create Date: 2024-10-15 19:39:27.381006

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '582c0f14a951'
down_revision = '44c14d50b0ef'
branch_labels = None
depends_on = None

def upgrade():
    # Create a new table with the desired structure
    op.create_table('new_spell_stats',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('version', sa.String(), nullable=False),
        sa.Column('champion', sa.String(), nullable=False),
        sa.Column('spell_id', sa.String(), nullable=False),
        sa.Column('spell_name', sa.String(), nullable=False),
        sa.Column('damage_type', sa.String()),
        sa.Column('damage_values', sa.JSON()),
        sa.Column('damage_ratios', sa.JSON()),
        sa.Column('max_rank', sa.Integer()),
        sa.Column('cooldown', sa.JSON()),
        sa.Column('cost', sa.JSON()),
        sa.Column('effect', sa.JSON()),
        sa.Column('range', sa.JSON()),
        sa.Column('resource', sa.String()),
        sa.Column('is_passive', sa.Boolean(), default=False)
    )

    # Get existing columns from the current spell_stats table
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing_columns = [col['name'] for col in inspector.get_columns('spell_stats')]

    # Construct the SELECT part of the INSERT statement dynamically
    select_cols = ', '.join(col for col in existing_columns if col in [
        'version', 'champion', 'spell_id', 'spell_name', 'damage_type', 'damage_values',
        'damage_ratios', 'max_rank', 'cooldown', 'cost', 'effect', 'range', 'resource', 'is_passive'
    ])

    # Copy data from the old table to the new one
    if select_cols:
        op.execute(f'''
            INSERT INTO new_spell_stats ({select_cols})
            SELECT {select_cols}
            FROM spell_stats
        ''')

    # Drop the old table and rename the new one
    op.drop_table('spell_stats')
    op.rename_table('new_spell_stats', 'spell_stats')

def downgrade():
    # Create a new table with the old structure
    op.create_table('old_spell_stats',
        sa.Column('version', sa.String(), nullable=False),
        sa.Column('champion', sa.String(), nullable=False),
        sa.Column('spell_id', sa.String(), nullable=False),
        sa.Column('spell_name', sa.String(), nullable=False)
    )

    # Copy data back to the old structure
    op.execute('''
        INSERT INTO old_spell_stats (version, champion, spell_id, spell_name)
        SELECT version, champion, spell_id, spell_name
        FROM spell_stats
    ''')

    # Drop the new table and rename the old one
    op.drop_table('spell_stats')
    op.rename_table('old_spell_stats', 'spell_stats')