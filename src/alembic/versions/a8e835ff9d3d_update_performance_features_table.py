"""update_performance_features_table

Revision ID: a8e835ff9d3d
Revises: 8d8676f499d9
Create Date: 2024-01-21
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'a8e835ff9d3d'
down_revision = '8d8676f499d9'
branch_labels = None
depends_on = None

def upgrade():
    # Drop the old table if it exists
    op.execute('DROP TABLE IF EXISTS performance_features')
    
    # Create the new table with constraints included
    op.create_table('performance_features',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
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
        sa.Column('champion_role_patch', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['participant_id'], ['participants.participant_id'], 
                              name='fk_performance_features_participant_id'),
        sa.UniqueConstraint('participant_id', name='uq_performance_features_participant_id')
    )

def downgrade():
    op.drop_table('performance_features')