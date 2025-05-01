import json
from dataclasses import dataclass
from typing import Any, Dict, List


from .utils import StrPath


@dataclass(frozen=True)
class SpiderDatum:
    schema_name: str
    query: str
    utterance: str

    @staticmethod
    def from_json(datum_json: Dict[str, Any]) -> "SpiderDatum":
        return SpiderDatum(
            schema_name=datum_json["db_id"],
            utterance=datum_json["question"],
            query=datum_json["query"],
        )


def load_spider_data(data_filepath: StrPath) -> List[SpiderDatum]:
    with open(data_filepath, encoding="utf-8") as f:
        json_data = json.load(f)
        return [SpiderDatum.from_json(d) for d in json_data]
