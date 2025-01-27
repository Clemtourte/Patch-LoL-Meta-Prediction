"""add_winrate_table

Revision ID: 9a0071ba6314
Revises: aecdb979ea32
Create Date: 2025-01-26 23:21:50.202706

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '9a0071ba6314'
down_revision = 'aecdb979ea32'
branch_labels = None
depends_on = None

def upgrade():
    # First drop table if exists
    op.execute("DROP TABLE IF EXISTS champion_winrates")
    op.create_table(
        'champion_winrates',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('patch', sa.String, nullable=False),
        sa.Column('champion_name', sa.String, nullable=False),
        sa.Column('winrate', sa.Float, nullable=False),
        sa.Column('pickrate', sa.Float, nullable=False),
        sa.Column('banrate', sa.Float, nullable=False),
        sa.Column('role_rate', sa.Float, nullable=True),
        sa.Column('total_games', sa.Integer, nullable=True)
    )
    
    op.create_index(
        'winrate_patch_champion_idx',  # Changed this line
        'champion_winrates',
        ['patch', 'champion_name']
    )

def downgrade():
    op.drop_index('winrate_patch_champion_idx', 'champion_winrates')  # Changed this line too
    op.drop_table('champion_winrates')