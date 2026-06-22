import csv
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .utils import StrPath


_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[\"`']?([A-Za-z0-9_]+)[\"`']?\s*\((.*)\)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class Column:
    name: str
    tpe: str
    nl_name: str = ""


@dataclass
class Table:
    name: str
    columns: List[Column]
    ddl: str = ""
    nl_name: str = ""


@dataclass
class DbSchema:
    name: str
    tables: List[Table] = field(default_factory=list)
    # Optional location of the sqlite file on disk
    sqlite_path: Optional[Path] = None


def _parse_column_defs(body: str) -> List[Column]:
    """Best-effort parse of column definitions from a CREATE TABLE body."""
    parts: List[str] = []
    depth = 0
    buf: List[str] = []
    for ch in body:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())

    columns: List[Column] = []
    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue
        upper = stripped.upper()
        if upper.startswith(
            (
                "PRIMARY KEY",
                "FOREIGN KEY",
                "UNIQUE",
                "CHECK",
                "CONSTRAINT",
                "INDEX",
                "KEY ",
            )
        ):
            continue
        tokens = stripped.split()
        if len(tokens) < 2:
            continue
        name = tokens[0].strip('"`\'')
        tpe = tokens[1].strip(",;")
        columns.append(Column(name=name, tpe=tpe, nl_name=name))
    return columns


def parse_ddl_csv(ddl_csv_path: StrPath) -> List[Table]:
    """Parse a Spider 2 ``DDL.csv`` file into a list of :class:`Table` objects.

    The CSV has two columns: ``table_name`` and ``DDL``.
    """
    tables: List[Table] = []
    with open(ddl_csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("table_name") or "").strip()
            ddl = row.get("DDL") or ""
            if not name:
                continue
            columns: List[Column] = []
            match = _CREATE_TABLE_RE.search(ddl)
            if match:
                columns = _parse_column_defs(match.group(2))
            tables.append(Table(name=name, columns=columns, ddl=ddl, nl_name=name))
    return tables


def _columns_from_sqlite(sqlite_path: Path) -> Dict[str, List[Column]]:
    columns_by_table: Dict[str, List[Column]] = {}
    with sqlite3.connect(str(sqlite_path)) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'"
        )
        for (table_name,) in cur.fetchall():
            cols: List[Column] = []
            for row in cur.execute(f'PRAGMA table_info("{table_name}")'):
                # row = (cid, name, type, notnull, dflt_value, pk)
                cols.append(
                    Column(name=row[1], tpe=row[2] or "", nl_name=row[1])
                )
            columns_by_table[table_name] = cols
    return columns_by_table


def load_schema(db_dir: StrPath, sqlite_path: Optional[StrPath] = None) -> DbSchema:
    """Load a single Spider 2 SQLite database schema.

    Args:
        db_dir: Directory containing ``DDL.csv`` (and optionally other resources).
        sqlite_path: Path to the ``.sqlite`` file backing this database.  When
            ``DDL.csv`` is missing we fall back to introspecting the sqlite file.
    """
    db_dir = Path(db_dir)
    ddl_path = db_dir / "DDL.csv"

    if ddl_path.exists():
        tables = parse_ddl_csv(ddl_path)
        if sqlite_path is not None and Path(sqlite_path).exists():
            introspected = _columns_from_sqlite(Path(sqlite_path))
            for table in tables:
                if not table.columns and table.name in introspected:
                    table.columns = introspected[table.name]
    elif sqlite_path is not None and Path(sqlite_path).exists():
        introspected = _columns_from_sqlite(Path(sqlite_path))
        tables = [
            Table(name=name, columns=cols, ddl="", nl_name=name)
            for name, cols in introspected.items()
        ]
    else:
        tables = []

    return DbSchema(
        name=db_dir.name,
        tables=tables,
        sqlite_path=Path(sqlite_path) if sqlite_path is not None else None,
    )


def load_schemas(
    schemas_dir: StrPath, sqlite_db_dir: Optional[StrPath] = None
) -> Dict[str, DbSchema]:
    """Load all schemas under ``schemas_dir``.

    Args:
        schemas_dir: A directory whose subdirectories each describe one
            database (the convention used by Spider 2-Lite at
            ``resource/databases/sqlite``).
        sqlite_db_dir: Directory containing ``{db}.sqlite`` files
            (``resource/databases/spider2-localdb`` in the public release).
    """
    schemas_dir = Path(schemas_dir)
    sqlite_dir = Path(sqlite_db_dir) if sqlite_db_dir is not None else None

    schemas: Dict[str, DbSchema] = {}
    if not schemas_dir.exists():
        return schemas

    for entry in sorted(schemas_dir.iterdir()):
        if not entry.is_dir():
            continue
        sqlite_path = None
        if sqlite_dir is not None:
            candidate = sqlite_dir / f"{entry.name}.sqlite"
            if candidate.exists():
                sqlite_path = candidate
        schemas[entry.name] = load_schema(entry, sqlite_path=sqlite_path)
    return schemas
