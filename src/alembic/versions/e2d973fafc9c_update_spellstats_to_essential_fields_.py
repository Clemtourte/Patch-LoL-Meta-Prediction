"""Update SpellStats to essential fields only

Revision ID: e2d973fafc9c
Revises: d5e194f1991f
Create Date: 2025-03-06 20:11:56.398314

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite
from sqlalchemy.engine import reflection

# revision identifiers, used by Alembic.
revision = 'e2d973fafc9c'
down_revision = 'd5e194f1991f'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop any leftover temporary table from a previous incomplete migration.
    op.execute("DROP TABLE IF EXISTS _alembic_tmp_participants")

    # Inspect the 'participants' table to check for an existing foreign key.
    conn = op.get_bind()
    insp = reflection.Inspector.from_engine(conn)
    fkeys = insp.get_foreign_keys("participants")
    existing_fk_names = [fk["name"] for fk in fkeys if fk.get("name")]

    with op.batch_alter_table('participants') as batch_op:
        if 'fk_participants_team_id' in existing_fk_names:
            batch_op.drop_constraint('fk_participants_team_id', type_='foreignkey')
        batch_op.create_foreign_key('fk_participants_team_id', 'teams', ['team_id'], ['team_id'])
    
    with op.batch_alter_table('spell_stats') as batch_op:
        batch_op.add_column(sa.Column('spell_type', sa.String(), nullable=False))
        batch_op.drop_index('idx_spell_stats_standardized_id')
        batch_op.create_index('idx_spell_stats_champ_spell', ['champion', 'spell_type', 'version'], unique=False)
        batch_op.drop_column('spell_id')
        batch_op.drop_column('effect_values')
        batch_op.drop_column('is_passive')
        batch_op.drop_column('standardized_id')


def downgrade() -> None:
    with op.batch_alter_table('spell_stats') as batch_op:
        batch_op.add_column(sa.Column('standardized_id', sa.String(length=100), nullable=True))
        batch_op.add_column(sa.Column('is_passive', sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column('effect_values', sqlite.JSON(), nullable=True))
        batch_op.add_column(sa.Column('spell_id', sa.String(), nullable=False))
        batch_op.drop_index('idx_spell_stats_champ_spell')
        batch_op.create_index('idx_spell_stats_standardized_id', ['standardized_id'], unique=False)
        batch_op.drop_column('spell_type')
    
    op.execute("DROP TABLE IF EXISTS _alembic_tmp_participants")
    conn = op.get_bind()
    insp = reflection.Inspector.from_engine(conn)
    fkeys = insp.get_foreign_keys("participants")
    existing_fk_names = [fk["name"] for fk in fkeys if fk.get("name")]
    
    with op.batch_alter_table('participants') as batch_op:
        if 'fk_participants_team_id' in existing_fk_names:
            batch_op.drop_constraint('fk_participants_team_id', type_='foreignkey')
        batch_op.create_foreign_key('fk_participants_team_id', 'teams', ['team_id'], ['team_id'], ondelete='CASCADE')
