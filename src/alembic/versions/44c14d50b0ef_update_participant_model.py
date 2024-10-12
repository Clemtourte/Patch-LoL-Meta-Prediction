"""update participant model

Revision ID: 44c14d50b0ef
Revises: 07eb7628af35
Create Date: 2023-10-11 14:17:52.736880

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '44c14d50b0ef'
down_revision = '07eb7628af35'
branch_labels = None
depends_on = None

def upgrade():
    # Drop existing tables
    op.drop_table('performance_features')
    op.drop_table('participants')
    op.drop_table('teams')
    op.drop_table('matches')

    # Recreate tables with new schema
    op.create_table('matches',
        sa.Column('match_id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('game_id', sa.String(), nullable=False),
        sa.Column('game_duration', sa.String(), nullable=False),
        sa.Column('patch', sa.String(), nullable=False),
        sa.Column('timestamp', sa.String(), nullable=False),
        sa.Column('mode', sa.String(), nullable=False),
        sa.Column('queue_id', sa.Integer(), nullable=False),
        sa.Column('platform', sa.String(), nullable=False),
        sa.PrimaryKeyConstraint('match_id'),
        sa.UniqueConstraint('game_id')
    )

    op.create_table('teams',
        sa.Column('team_id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('match_id', sa.Integer(), nullable=False),
        sa.Column('team_name', sa.String(), nullable=False),
        sa.Column('win', sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(['match_id'], ['matches.match_id'], ),
        sa.PrimaryKeyConstraint('team_id')
    )

    op.create_table('participants',
        sa.Column('participant_id', sa.Integer(), autoincrement=True, nullable=False),
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
        sa.ForeignKeyConstraint(['team_id'], ['teams.team_id'], ),
        sa.PrimaryKeyConstraint('participant_id')
    )

    op.create_table('performance_features',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('participant_id', sa.Integer(), nullable=False),
        sa.Column('kill_participation', sa.Float(), nullable=True),
        sa.Column('death_share', sa.Float(), nullable=True),
        sa.Column('damage_share', sa.Float(), nullable=True),
        sa.Column('damage_taken_share', sa.Float(), nullable=True),
        sa.Column('gold_share', sa.Float(), nullable=True),
        sa.Column('heal_share', sa.Float(), nullable=True),
        sa.Column('damage_mitigated_share', sa.Float(), nullable=True),
        sa.Column('cs_share', sa.Float(), nullable=True),
        sa.Column('vision_share', sa.Float(), nullable=True),
        sa.Column('vision_denial_share', sa.Float(), nullable=True),
        sa.Column('xp_share', sa.Float(), nullable=True),
        sa.Column('cc_share', sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(['participant_id'], ['participants.participant_id'], ),
        sa.PrimaryKeyConstraint('id')
    )

def downgrade():
    op.drop_table('performance_features')
    op.drop_table('participants')
    op.drop_table('teams')
    op.drop_table('matches')