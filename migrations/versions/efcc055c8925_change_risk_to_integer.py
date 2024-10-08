"""Change risk to integer

Revision ID: efcc055c8925
Revises: e23ff17f55eb
Create Date: 2024-08-30 00:06:55.401836

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'efcc055c8925'
down_revision = 'e23ff17f55eb'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('ingredient', schema=None) as batch_op:
        batch_op.alter_column('risk',
                              existing_type=sa.VARCHAR(length=255),
                              type_=sa.Integer(),
                              existing_nullable=False,
                              existing_server_default=None,
                              postgresql_using='risk::INTEGER')

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('ingredient', schema=None) as batch_op:
        batch_op.alter_column('risk',
                              existing_type=sa.Integer(),
                              type_=sa.VARCHAR(length=255),
                              existing_nullable=False,
                              existing_server_default=None,
                              postgresql_using='risk::VARCHAR(255)')

    # ### end Alembic commands ###
