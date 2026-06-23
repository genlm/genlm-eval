import os  # pylint: disable=unused-import
from typing import Union

StrPath = Union[str, "os.PathLike[str]"]


def backend_for_instance(instance_id: str) -> str:
    """Map a Spider 2.0-Lite ``instance_id`` to its database backend.

    The backend is encoded in the id prefix, following the official suite's
    convention: ``sf_bq*`` (and any ``sf*``) is Snowflake, ``bq*``/``ga*`` is
    BigQuery, and everything else (``local*``) is SQLite.  ``sf`` is checked
    before ``bq`` because Snowflake ids look like ``sf_bq001``.
    """
    if instance_id.startswith("sf"):
        return "snowflake"
    if instance_id.startswith(("bq", "ga")):
        return "bigquery"
    return "sqlite"


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
