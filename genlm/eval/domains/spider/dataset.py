from typing import List
from pathlib import Path
from pydantic import BaseModel

from .spider.utils import serialize_schema
from .spider.dialogue import load_spider_data
from .spider.schema import load_schemas

from genlm.eval.core import Dataset


class SpiderInstance(BaseModel):
    """Schema for text to SQL instance."""

    utterance: str
    schema_name: str
    gold: str
    id: int
    schema_str: str
    few_shot_examples: List[tuple]
    tables: List

    def __str__(self):
        return f"utterance: {self.utterance}, schema_name: {self.schema_name} (id: {self.id})"


class SpiderDataset(Dataset[SpiderInstance]):
    """Dataset for text to SQL evaluation."""

    def __init__(self, dev_data, spider_schemas, train_data):
        self.dev_data = dev_data
        self.spider_schemas = spider_schemas

        self.few_shot_examples = []
        for example_id in [10, 100, 1000]:
            train_datum = train_data[example_id]
            self.few_shot_examples.append(
                (
                    serialize_schema(self.spider_schemas[train_datum.schema_name]),
                    train_datum.utterance,
                    train_datum.query,
                )
            )

    @classmethod
    def from_spider_dir(cls, raw_spider_dir):
        raw_spider_dir = Path(raw_spider_dir)
        dev_data = load_spider_data(raw_spider_dir / "dev.json")
        spider_schemas = load_schemas(
            schemas_path=raw_spider_dir / "tables.json",
            db_path=raw_spider_dir / "database",
        )
        train_data = load_spider_data(raw_spider_dir / "train_spider.json")
        return cls(dev_data, spider_schemas, train_data)

    def __iter__(self):
        for instance_id, dev_datum in enumerate(self.dev_data):
            yield SpiderInstance(
                schema_name=dev_datum.schema_name,
                schema_str=serialize_schema(self.spider_schemas[dev_datum.schema_name]),
                utterance=dev_datum.utterance,
                gold=dev_datum.query,
                id=instance_id,
                few_shot_examples=self.few_shot_examples,
                tables=self.spider_schemas[dev_datum.schema_name].tables,
            )

    @property
    def schema(self):
        return SpiderInstance
