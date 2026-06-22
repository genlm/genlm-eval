import json
from functools import lru_cache
from pathlib import Path
from typing import List, Union

from .spider2_eval.dialogue import (
    Spider2Datum,
    load_external_knowledge,
    load_spider2_data,
)
from .spider2_eval.evaluator import Evaluator as BaseSpider2Evaluator
from .spider2_eval.schema import load_schemas
from .spider2_eval.utils import serialize_schema

from genlm.eval.core import Dataset, EvaluationResult, Evaluator, Instance
from genlm.eval.util import chat_template_messages


###########
# Dataset #
###########


class Spider2Instance(Instance):
    """Schema for a Spider 2.0-Lite text-to-SQL instance."""

    utterance: str
    schema_name: str
    gold: str
    schema_str: str
    lark_grammar: Union[str, None]
    few_shot_examples: List[tuple]
    tables: List
    user_message: str
    spider2_instance_id: str
    external_knowledge: Union[str, None]

    def __str__(self):
        return (
            f"utterance: {self.utterance}, "
            f"schema_name: {self.schema_name} (id: {self.instance_id}, "
            f"spider2_id: {self.spider2_instance_id})"
        )


class Spider2Dataset(Dataset[Spider2Instance]):
    """Dataset for Spider 2.0-Lite text-to-SQL evaluation.

    Spider 2.0-Lite stores each instance as a JSONL row with the fields
    ``instance_id``, ``db``, ``question`` and ``external_knowledge``.  The gold
    SQL lives in ``evaluation_suite/gold/sql/{instance_id}.sql`` and the SQLite
    databases under ``resource/databases/spider2-localdb/``.

    By default the dataset iterates over the *local* (SQLite-backed)
    instances since those are the ones the bundled execution evaluator can
    score.  Pass an ``instance_filter`` to iterate over a different slice.
    """

    def __init__(
        self,
        dev_data,
        spider2_schemas,
        train_data,
        documents_dir=None,
        grammars=None,
        few_shot_example_ids=None,
    ):
        self.dev_data: List[Spider2Datum] = dev_data
        self.train_data: List[Spider2Datum] = train_data
        self.spider2_schemas = spider2_schemas
        self.documents_dir = Path(documents_dir) if documents_dir is not None else None

        if few_shot_example_ids is None:
            few_shot_example_ids = []  # pragma: no cover

        self.few_shot_examples = []
        for example_id in few_shot_example_ids:
            train_datum = train_data[example_id]
            schema = spider2_schemas.get(train_datum.schema_name)
            schema_str = serialize_schema(schema) if schema is not None else ""
            self.few_shot_examples.append(
                (
                    self.user_message_template(
                        schema_str,
                        train_datum.utterance,
                        external_knowledge=load_external_knowledge(
                            self.documents_dir, train_datum.external_knowledge
                        )
                        if self.documents_dir is not None
                        else None,
                    ),
                    train_datum.query,
                )
            )

        self.grammars = grammars if grammars else {}

    @staticmethod
    def user_message_template(schema_str, utterance, external_knowledge=None):
        extra = ""
        if external_knowledge:
            extra = (
                "\nAdditional context to ground the question:\n"
                f"{external_knowledge.strip()}\n"
            )
        return (
            "Here is a database schema:\n"
            f"{schema_str}\n"
            f"{extra}"
            "Please write me a SQL statement that answers the following question:\n"
            f"{utterance}\n"
            "Remember, DO NOT provide any commentary or explanation of what the code does, "
            "just the SQL statement ending in a semicolon."
        )

    @classmethod
    def from_spider2_dir(
        cls,
        raw_spider2_dir,
        grammar_json_path=None,
        train_jsonl_path=None,
        instance_filter=None,
        **kwargs,
    ):
        """Build a dataset from a Spider 2.0-Lite checkout.

        Args:
            raw_spider2_dir: Path to the ``spider2-lite`` directory.
            grammar_json_path: Optional path to a JSON mapping
                ``db -> lark grammar`` (mirrors Spider 1's ``grammars.json``).
            train_jsonl_path: Optional path to a separate JSONL with training /
                few-shot instances.  Defaults to the dev JSONL when omitted.
            instance_filter: Optional callable ``instance_id -> bool``.  By
                default we keep instances whose ids start with ``local`` since
                those are SQLite-backed and runnable end-to-end.
        """
        raw_spider2_dir = Path(raw_spider2_dir)
        dev_jsonl = raw_spider2_dir / "spider2-lite.jsonl"
        gold_sql_dir = raw_spider2_dir / "evaluation_suite" / "gold" / "sql"
        documents_dir = raw_spider2_dir / "resource" / "documents"

        if instance_filter is None:
            instance_filter = lambda iid: iid.startswith("local")  # noqa: E731

        dev_data = load_spider2_data(
            dev_jsonl, gold_sql_dir=gold_sql_dir, instance_filter=instance_filter
        )

        if train_jsonl_path is not None:
            train_data = load_spider2_data(
                train_jsonl_path,
                gold_sql_dir=gold_sql_dir,
                instance_filter=instance_filter,
            )
        else:
            train_data = list(dev_data)

        spider2_schemas = load_schemas(
            schemas_dir=raw_spider2_dir / "resource" / "databases" / "sqlite",
            sqlite_db_dir=raw_spider2_dir / "resource" / "databases" / "spider2-localdb",
        )

        if grammar_json_path is None:
            grammars = None
        else:
            with open(grammar_json_path, "r") as f:
                grammars = json.load(f)

        return cls(
            dev_data,
            spider2_schemas,
            train_data,
            documents_dir=documents_dir if documents_dir.exists() else None,
            grammars=grammars,
            **kwargs,
        )

    def __iter__(self):
        for instance_id, dev_datum in enumerate(self.dev_data):
            schema = self.spider2_schemas.get(dev_datum.schema_name)
            schema_str = serialize_schema(schema) if schema is not None else ""
            external_knowledge_text = (
                load_external_knowledge(self.documents_dir, dev_datum.external_knowledge)
                if self.documents_dir is not None
                else None
            )
            yield Spider2Instance(
                schema_name=dev_datum.schema_name,
                schema_str=schema_str,
                lark_grammar=self.grammars.get(dev_datum.schema_name),
                utterance=dev_datum.utterance,
                gold=dev_datum.query,
                instance_id=instance_id,
                spider2_instance_id=dev_datum.instance_id,
                external_knowledge=external_knowledge_text,
                few_shot_examples=self.few_shot_examples,
                tables=schema.tables if schema is not None else [],
                user_message=self.user_message_template(
                    schema_str,
                    dev_datum.utterance,
                    external_knowledge=external_knowledge_text,
                ),
            )

    @property
    def schema(self):
        return Spider2Instance


#############
# Evaluator #
#############


class Spider2Evaluator(Evaluator[Spider2Instance]):
    """Execution-based evaluator for Spider 2.0-Lite (SQLite backend)."""

    def __init__(
        self,
        raw_spider2_dir,
        sqlite_dir=None,
        exec_result_dir=None,
        eval_config_path=None,
        evaluator_timeout=None,
    ):
        self.raw_spider2_dir = Path(raw_spider2_dir)
        self.evaluator = BaseSpider2Evaluator(
            self.raw_spider2_dir,
            sqlite_dir=sqlite_dir,
            exec_result_dir=exec_result_dir,
            eval_config_path=eval_config_path,
            timeout=evaluator_timeout,
        )

    @lru_cache
    def cached_eval(self, pred, gold, db, instance_id):
        return self.evaluator.evaluate(pred, gold, db_name=db, instance_id=instance_id)

    def evaluate_sample(self, instance, response):
        is_correct, reason, level = self.cached_eval(
            response,
            instance.gold,
            instance.schema_name,
            instance.spider2_instance_id,
        )
        if reason is None:
            reason = "valid"
        return EvaluationResult(
            score=float(is_correct),
            desc=reason,
            metadata={"level": level} if level is not None else {},
        )


###############
# Model Utils #
###############


SYSTEM_PROMPT = (
    "You are a coding assistant helping an analyst answer questions over business data in SQL. "
    "More specifically, the analyst provides you a database schema "
    "(tables in the database along with their column names and types) "
    "and asks a complex question about the data that can be solved by issuing a SQL query. "
    "In response, you write the SQL statement that answers the question. "
    "You do not provide any commentary or explanation of what the code does, "
    "just the SQL statement ending in a semicolon."
)


def default_prompt_formatter(
    tokenizer,
    instance,
    use_chat_format=True,
    system_prompt=SYSTEM_PROMPT,
):
    """Default prompt formatter for Spider 2.0-Lite.

    Args:
        tokenizer: The tokenizer to use.
        instance (Spider2Instance): The instance to format.
        use_chat_format (bool): Whether to use chat format.
        system_prompt (str): The system prompt to use.

    Returns:
        (list[int]): The prompt ids.
    """
    if use_chat_format:
        return tokenizer.apply_chat_template(
            conversation=chat_template_messages(
                system_prompt,
                instance.few_shot_examples,
                instance.user_message,
            ),
            tokenize=True,
            add_generation_prompt=True,
        )
    else:
        return tokenizer.encode(
            (
                system_prompt
                + "\n\n"
                + "\n\n".join(
                    f"{input}\nSQL query: {output}"
                    for input, output in instance.few_shot_examples
                )
                + "\n\n"
                + instance.user_message
                + "\nSQL query:"
            )
        )
