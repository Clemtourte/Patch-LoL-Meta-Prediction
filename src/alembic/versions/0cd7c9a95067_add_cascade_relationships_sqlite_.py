"""Add cascade relationships sqlite compatible

Revision ID: 0cd7c9a95067
Revises: a8e835ff9d3d
Create Date: 2025-01-14 11:00:31.966246

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision = '0cd7c9a95067'
down_revision = 'a8e835ff9d3d'
branch_labels = None
depends_on = None

def upgrade():
    # First drop existing tables safely
    connection = op.get_bind()
    connection.execute(text('DROP TABLE IF EXISTS teams_new'))
    connection.execute(text('DROP TABLE IF EXISTS participants_new'))
    
    # Create new tables
    op.create_table(
        'teams_new',
        sa.Column('team_id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('match_id', sa.Integer(), nullable=False),
        sa.Column('team_name', sa.String(), nullable=False),
        sa.Column('win', sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(['match_id'], ['matches.match_id'], ondelete='CASCADE')
    )

    op.create_table(
        'participants_new',
        sa.Column('participant_id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('team_id', sa.Integer(), nullable=False),
        sa.Column('summoner_id', sa.String(), nullable=False),
        sa.Column('summoner_name', sa.String(), nullable=False),
        sa.Column('champion_name', sa.String(), nullable=False),
        sa.Column('champion_id', sa.Integer(), nullable=False),
        sa.Column('champ_level', sa.Integer(), nullable=False),
        sa.Column('role', sa.String(), nullable=False),
        sa.Column('lane', sa.String(), nullable=False),
        sa.Column('position', sa.String(), nullable=False),
        sa.Column('kills', sa.Integer(), nullable=False),
        sa.Column('deaths', sa.Integer(), nullable=False),
        sa.Column('assists', sa.Integer(), nullable=False),
        sa.Column('kda', sa.Float(), nullable=False),
        sa.Column('gold_earned', sa.Integer(), nullable=False),
        sa.Column('total_damage_dealt', sa.Integer(), nullable=False),
        sa.Column('cs', sa.Integer(), nullable=False),
        sa.Column('total_heal', sa.Integer(), nullable=True),
        sa.Column('damage_taken', sa.Integer(), nullable=True),
        sa.Column('damage_mitigated', sa.Integer(), nullable=True),
        sa.Column('wards_placed', sa.Integer(), nullable=True),
        sa.Column('wards_killed', sa.Integer(), nullable=True),
        sa.Column('time_ccing_others', sa.Integer(), nullable=True),
        sa.Column('xp', sa.Integer(), nullable=True),
        sa.Column('performance_score', sa.Float(), nullable=True),
        sa.Column('standardized_performance_score', sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(['team_id'], ['teams_new.team_id'], ondelete='CASCADE')
    )

    # Copy data
    connection.execute(text(
        'INSERT INTO teams_new (team_id, match_id, team_name, win) SELECT team_id, match_id, team_name, win FROM teams'
    ))
    connection.commit()

    connection.execute(text(
        'INSERT INTO participants_new SELECT * FROM participants'
    ))
    connection.commit()

    # Drop old tables
    connection.execute(text('DROP TABLE IF EXISTS participants'))
    connection.execute(text('DROP TABLE IF EXISTS teams'))
    connection.commit()

    # Rename new tables
    op.rename_table('teams_new', 'teams')
    op.rename_table('participants_new', 'participants')

def downgrade():
    connection = op.get_bind()

    op.drop_table('participants')
    op.drop_table('teams')

    op.create_table(
        'teams',
        sa.Column('team_id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('match_id', sa.Integer(), nullable=False),
        sa.Column('team_name', sa.String(), nullable=False),
        sa.Column('win', sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(['match_id'], ['matches.match_id'])
    )

    op.create_table(
        'participants',
        sa.Column('participant_id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('team_id', sa.Integer(), nullable=False),
        sa.Column('summoner_id', sa.String(), nullable=False),
        sa.Column('summoner_name', sa.String(), nullable=False),
        sa.Column('champion_name', sa.String(), nullable=False),
        sa.Column('champion_id', sa.Integer(), nullable=False),
        sa.Column('champ_level', sa.Integer(), nullable=False),
        sa.Column('role', sa.String(), nullable=False),
        sa.Column('lane', sa.String(), nullable=False),
        sa.Column('position', sa.String(), nullable=False),
        sa.Column('kills', sa.Integer(), nullable=False),
        sa.Column('deaths', sa.Integer(), nullable=False),
        sa.Column('assists', sa.Integer(), nullable=False),
        sa.Column('kda', sa.Float(), nullable=False),
        sa.Column('gold_earned', sa.Integer(), nullable=False),
        sa.Column('total_damage_dealt', sa.Integer(), nullable=False),
        sa.Column('cs', sa.Integer(), nullable=False),
        sa.Column('total_heal', sa.Integer(), nullable=True),
        sa.Column('damage_taken', sa.Integer(), nullable=True),
        sa.Column('damage_mitigated', sa.Integer(), nullable=True),
        sa.Column('wards_placed', sa.Integer(), nullable=True),
        sa.Column('wards_killed', sa.Integer(), nullable=True),
        sa.Column('time_ccing_others', sa.Integer(), nullable=True),
        sa.Column('xp', sa.Integer(), nullable=True),
        sa.Column('performance_score', sa.Float(), nullable=True),
        sa.Column('standardized_performance_score', sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(['team_id'], ['teams.team_id'])
    )

    connection.execute(text(
        'INSERT INTO teams (team_id, match_id, team_name, win) SELECT team_id, match_id, team_name, win FROM teams_new'
    ))
    connection.commit()

    connection.execute(text(
        'INSERT INTO participants SELECT * FROM participants_new'
    ))
    connection.commit()

    op.drop_table('participants_new')
    op.drop_table('teams_new')