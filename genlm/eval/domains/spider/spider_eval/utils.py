import os  # pylint: disable=unused-import
from typing import Union

# This can be used to annotate arguments that are supposed to be file paths.
StrPath = Union[str, "os.PathLike[str]"]


def serialize_schema(db_schema):
    table_strs = []
    for table in db_schema.tables:
        column_strs = []
        for column in table.columns:
            column_strs.append(
                f"* {column.name} ({column.tpe.value}): {column.nl_name}"
            )
        table_str = "\n".join([table.name] + column_strs)
        table_strs.append(table_str)

    return "\n\n".join(table_strs)
