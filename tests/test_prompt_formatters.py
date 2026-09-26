import pytest
from transformers import AutoTokenizer

from genlm.eval.domains.goal_inference import (
    GoalInferenceInstance,
    goal_default_prompt_formatter,
)
from genlm.eval.domains.json_schema import (
    JSONSchemaBenchInstance,
    default_prompt_formatter as json_schema_prompt_formatter,
)
from genlm.eval.domains.pattern_matching import (
    PatternMatchingInstance,
    default_prompt_formatter as pattern_matching_prompt_formatter,
)
from genlm.eval.domains.spider import (
    SpiderInstance,
    default_prompt_formatter as spider_prompt_formatter,
)

CHAT_FORMATTERS = [
    pytest.param(
        spider_prompt_formatter,
        SpiderInstance(
            instance_id=0,
            utterance="How many singers are there?",
            schema_name="concert_singer",
            gold="SELECT count(*) FROM singer",
            schema_str="",
            lark_grammar=None,
            few_shot_examples=[("How many concerts?", "SELECT count(*) FROM concert;")],
            tables=[],
            user_message="How many singers are there?",
        ),
        id="spider",
    ),
    pytest.param(
        pattern_matching_prompt_formatter,
        PatternMatchingInstance(instance_id=0, pattern="a|b"),
        id="pattern_matching",
    ),
    pytest.param(
        json_schema_prompt_formatter,
        JSONSchemaBenchInstance(
            instance_id=0, task="Github_easy", json_schema={"type": "object"}
        ),
        id="json_schema",
    ),
    pytest.param(
        goal_default_prompt_formatter,
        GoalInferenceInstance(
            instance_id=0,
            nl_goal="Stack b1 on b2.",
            problem_text="",
            masked_pddl="",
            prefix_pddl="(define (problem p) (:domain blocksworld)",
            domain_name="blocksworld",
        ),
        id="goal_inference",
    ),
]


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")


@pytest.mark.network
@pytest.mark.parametrize("formatter, instance", CHAT_FORMATTERS)
def test_chat_format_returns_token_ids(tokenizer, formatter, instance):
    ids = formatter(tokenizer, instance, use_chat_format=True)
    assert isinstance(ids, list) and ids
    assert all(isinstance(i, int) for i in ids)
