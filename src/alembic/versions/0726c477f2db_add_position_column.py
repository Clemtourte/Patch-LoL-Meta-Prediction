"""add position column and modify champ_level

Revision ID: 0726c477f2db
Revises: 
Create Date: 2024-09-04 14:22:05.952153

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision: str = '0726c477f2db'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # Get the SQLAlchemy connection
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)
    existing_columns = [c['name'] for c in inspector.get_columns('participants')]

    # Add 'lane' column if it doesn't exist
    if 'lane' not in existing_columns:
        op.add_column('participants', sa.Column('lane', sa.String(), nullable=False, server_default=''))

    # Add 'position' column if it doesn't exist
    if 'position' not in existing_columns:
        op.add_column('participants', sa.Column('position', sa.String(), nullable=False, server_default=''))

    # For SQLite, we need to recreate the table to change column type
    with op.batch_alter_table('participants') as batch_op:
        batch_op.alter_column('champ_level',
                              existing_type=sa.NUMERIC(),
                              type_=sa.Integer(),
                              existing_nullable=False)

def downgrade() -> None:
    # For SQLite, we need to recreate the table to change column type
    with op.batch_alter_table('participants') as batch_op:
        batch_op.alter_column('champ_level',
                              existing_type=sa.Integer(),
                              type_=sa.NUMERIC(),
                              existing_nullable=False)
    
    # Get the SQLAlchemy connection
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)
    existing_columns = [c['name'] for c in inspector.get_columns('participants')]
    
    # Drop 'position' column if it exists
    if 'position' in existing_columns:
        with op.batch_alter_table('participants') as batch_op:
            batch_op.drop_column('position')
    
    # Drop 'lane' column if it exists
    if 'lane' in existing_columns:
        with op.batch_alter_table('participants') as batch_op:
            batch_op.drop_column('lane')