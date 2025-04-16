import os
from genlm.control import BoolCFG

from genlm.eval.models.control import ControlModelAdaptor


class SpiderModel(ControlModelAdaptor):
    system_prompt = (
        "You are a coding assistant helping an analyst answer questions over business data in SQL. "
        "More specifically, the analyst provides you a database schema "
        "(tables in the database along with their column names and types) "
        "and asks a question about the data that can be solved by issuing a SQL query to the database. "
        "In response, you write the SQL statement that answers the question. "
        "You do not provide any commentary or explanation of what the code does, "
        "just the SQL statement ending in a semicolon."
    )

    user_message_template = """Here is a database schema:

{schema_str}

Please write me a SQL statement that answers the following question: {utterance}

Remember, DO NOT provide any commentary or explanation of what the code does, just the SQL statement ending in a semicolon."""

    def __init__(self, *, grammar_dir=None, **kwargs):
        if grammar_dir is None:
            self.grammar_dir = os.path.join(os.path.dirname(__file__), "grammars")
        else:
            self.grammar_dir = grammar_dir
        super().__init__(**kwargs)

    def bool_cfg(self, db_name):
        return BoolCFG.from_lark(
            open(os.path.join(self.grammar_dir, db_name + ".lark"), "r").read()
        )

    def make_prompt_ids(self, instance):
        """Make the prompt ids for the model."""
        few_shot_examples = [
            (
                self.user_message_template.format(
                    schema_str=schema,
                    utterance=utterance,
                ),
                query,
            )
            for schema, utterance, query in instance.few_shot_examples
        ]
        return self.llm.model.tokenizer.apply_chat_template(
            self.chat_template_formatter(
                self.system_prompt,
                few_shot_examples,
                self.user_message_template.format(
                    schema_str=instance.schema_str,
                    utterance=instance.utterance,
                ),
            ),
            tokenize=True,
            add_generation_prompt=True,
        )
