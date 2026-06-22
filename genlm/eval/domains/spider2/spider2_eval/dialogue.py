import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils import StrPath


@dataclass(frozen=True)
class Spider2Datum:
    instance_id: str
    schema_name: str
    utterance: str
    query: str
    external_knowledge: Optional[str] = None

    @staticmethod
    def from_json(
        datum_json: Dict[str, Any],
        gold_sql: str,
    ) -> "Spider2Datum":
        return Spider2Datum(
            instance_id=datum_json["instance_id"],
            schema_name=datum_json["db"],
            utterance=datum_json["question"],
            query=gold_sql,
            external_knowledge=datum_json.get("external_knowledge"),
        )


def _read_text(path: Path) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def load_spider2_data(
    data_filepath: StrPath,
    gold_sql_dir: Optional[StrPath] = None,
    instance_filter=None,
) -> List[Spider2Datum]:
    """Load Spider 2.0-Lite instances from a JSONL file.

    Args:
        data_filepath: Path to ``spider2-lite.jsonl`` (one JSON object per line
            with keys ``instance_id``, ``db``, ``question``, ``external_knowledge``).
        gold_sql_dir: Optional directory containing ``{instance_id}.sql`` files.
            When provided, the gold SQL is attached to each datum; otherwise the
            ``query`` field is an empty string.
        instance_filter: Optional callable ``instance_id -> bool``.  Only
            instances for which the callable returns True are kept.
    """
    gold_dir = Path(gold_sql_dir) if gold_sql_dir is not None else None

    data: List[Spider2Datum] = []
    with open(data_filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            if instance_filter is not None and not instance_filter(obj["instance_id"]):
                continue

            gold_sql = ""
            if gold_dir is not None:
                gold_path = gold_dir / f"{obj['instance_id']}.sql"
                if gold_path.exists():
                    gold_sql = _read_text(gold_path)

            data.append(Spider2Datum.from_json(obj, gold_sql))
    return data


def load_external_knowledge(
    documents_dir: StrPath, document_name: Optional[str]
) -> Optional[str]:
    """Load an external knowledge document, returning its text content."""
    if not document_name:
        return None
    path = Path(documents_dir) / document_name
    if not path.exists():
        return None
    return _read_text(path)
