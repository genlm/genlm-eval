import os  # pylint: disable=unused-import
from typing import Union

StrPath = Union[str, "os.PathLike[str]"]


def serialize_schema(db_schema):
    """Render a Spider2 schema as a DDL listing.

    Each table is rendered as its CREATE TABLE statement (taken from
    ``DDL.csv``).  Tables are separated by a blank line.
    """
    table_strs = []
    for table in db_schema.tables:
        if table.ddl:
            table_strs.append(table.ddl.strip())
        else:
            column_strs = [
                f"* {column.name} ({column.tpe}): {column.name}" for column in table.columns
            ]
            table_strs.append("\n".join([table.name] + column_strs))

    return "\n\n".join(table_strs)
