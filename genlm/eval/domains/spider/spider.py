import json
from typing import List, Union
from pathlib import Path
from functools import lru_cache

from .spider_eval.utils import serialize_schema
from .spider_eval.dialogue import load_spider_data
from .spider_eval.schema import load_schemas
from .spider_eval.evaluator import Evaluator as BaseSpiderEvaluator

from genlm.eval.util import chat_template_messages
from genlm.eval.core import Evaluator, EvaluationResult, Instance, Dataset


###########
# Dataset #
###########


class SpiderInstance(Instance):
    """Schema for text to SQL instance."""

    utterance: str
    schema_name: str
    gold: str
    schema_str: str
    lark_grammar: Union[str, None]
    few_shot_examples: List[tuple]
    tables: List
    user_message: str

    def __str__(self):
        return f"utterance: {self.utterance}, schema_name: {self.schema_name} (id: {self.instance_id})"


class SpiderDataset(Dataset[SpiderInstance]):
    """Dataset for text to SQL evaluation."""

    def __init__(
        self,
        dev_data,
        spider_schemas,
        train_data,
        grammars=None,
        few_shot_example_ids=None,
    ):
        self.dev_data = dev_data
        self.spider_schemas = spider_schemas

        if few_shot_example_ids is None:
            few_shot_example_ids = [10, 100, 1000]  # pragma: no cover

        self.few_shot_examples = []
        for example_id in few_shot_example_ids:
            train_datum = train_data[example_id]
            self.few_shot_examples.append(
                (
                    self.user_message_template(
                        serialize_schema(self.spider_schemas[train_datum.schema_name]),
                        train_datum.utterance,
                    ),
                    train_datum.query,
                )
            )

        self.grammars = grammars if grammars else {}

    @staticmethod
    def user_message_template(schema_str, utterance):
        return (
            "Here is a database schema:\n"
            f"{schema_str}\n"
            "Please write me a SQL statement that answers the following question:\n"
            f"{utterance}\n"
            "Remember, DO NOT provide any commentary or explanation of what the code does, just the SQL statement ending in a semicolon."
        )

    @classmethod
    def from_spider_dir(cls, raw_spider_dir, grammar_json_path=None, **kwargs):
        raw_spider_dir = Path(raw_spider_dir)
        dev_data = load_spider_data(raw_spider_dir / "dev.json")
        spider_schemas = load_schemas(
            schemas_path=raw_spider_dir / "tables.json",
            db_path=raw_spider_dir / "database",
        )
        train_data = load_spider_data(raw_spider_dir / "train_spider.json")

        if grammar_json_path is None:
            grammars = None
        else:
            with open(grammar_json_path, "r") as f:
                grammars = json.load(f)

        return cls(dev_data, spider_schemas, train_data, grammars, **kwargs)

    def __iter__(self):
        for instance_id, dev_datum in enumerate(self.dev_data):
            schema_str = serialize_schema(self.spider_schemas[dev_datum.schema_name])
            yield SpiderInstance(
                schema_name=dev_datum.schema_name,
                schema_str=schema_str,
                lark_grammar=self.grammars.get(dev_datum.schema_name),
                utterance=dev_datum.utterance,
                gold=dev_datum.query,
                instance_id=instance_id,
                few_shot_examples=self.few_shot_examples,
                tables=self.spider_schemas[dev_datum.schema_name].tables,
                user_message=self.user_message_template(
                    schema_str,
                    dev_datum.utterance,
                ),
            )

    @property
    def schema(self):
        return SpiderInstance


#############
# Evaluator #
#############


class SpiderEvaluator(Evaluator[SpiderInstance]):
    """Evaluator for Spider."""

    def __init__(
        self,
        raw_spider_dir,
        evaluator_timeout=None,
    ):
        self.raw_spider_dir = Path(raw_spider_dir)
        self.evaluator = BaseSpiderEvaluator(
            self.raw_spider_dir, timeout=evaluator_timeout
        )

    @lru_cache
    def cached_eval(self, x, y, db):
        return self.evaluator.evaluate(x, y, db_name=db)

    def evaluate_sample(self, instance, response):
        is_correct, reason, level = self.cached_eval(
            response, instance.gold, instance.schema_name
        )
        if reason is None:
            reason = "valid"
        return EvaluationResult(
            score=float(is_correct), desc=reason, metadata={"level": level}
        )


###############
# Model Utils #
###############


SYSTEM_PROMPT = (
    "You are a coding assistant helping an analyst answer questions over business data in SQL. "
    "More specifically, the analyst provides you a database schema "
    "(tables in the database along with their column names and types) "
    "and asks a question about the data that can be solved by issuing a SQL query to the database. "
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
    """Default prompt formatter for pattern matching.

    Args:
        tokenizer (Tokenizer): The tokenizer to use.
        instance (SpiderInstance): The instance to format.
        use_chat_format (bool): Whether to use chat format.
        system_prompt (str): The system prompt to use.

    Returns:
        (list[int]): The prompt ids.
    """
    if use_chat_format:
        return tokenizer.apply_chat_template(
            messages=chat_template_messages(
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
