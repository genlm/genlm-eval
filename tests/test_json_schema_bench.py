import pytest
import json
from transformers import AutoTokenizer
from genlm.eval.domains.json_schema import (
    JSONSchemaBenchDataset,
    JSONSchemaBenchInstance,
    JSONSchemaBenchEvaluator,
    default_prompt_formatter,
    few_shots_messages_formatter,
    DEFAULT_SYSTEM_PROMPT,
)


@pytest.fixture
def dataset():
    return JSONSchemaBenchDataset.from_tasks(["Github_easy"])


TEST_SCHEMA = {
    "id": "https://raw.githubusercontent.com/OAI/OpenAPI-Specification/master/schemas/v1.2/infoObject.json#",
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "info object (section 5.1.3)",
    "type": "object",
    "required": ["title", "description"],
    "properties": {
        "title": {"type": "string"},
        "description": {"type": "string"},
        "termsOfServiceUrl": {"type": "string", "format": "uri"},
        "contact": {"type": "string", "format": "email"},
        "license": {"type": "string"},
        "licenseUrl": {"type": "string", "format": "uri"},
    },
    "additionalProperties": False,
}


@pytest.fixture
def test_instance():
    return JSONSchemaBenchInstance(
        json_schema=TEST_SCHEMA, instance_id=0, task="Github_easy"
    )


def test_dataset(dataset):
    assert dataset.schema is JSONSchemaBenchInstance

    assert len(dataset) > 1
    for instance in dataset:
        assert isinstance(instance, JSONSchemaBenchInstance)
        assert instance.task == "Github_easy"
        assert instance.json_schema is not None
        assert instance.instance_id is not None
        repr(instance)


def test_evaluator(test_instance):
    evaluator = JSONSchemaBenchEvaluator()

    # Test invalid json
    result = evaluator.evaluate_sample(test_instance, "not a json")
    assert result.score == 0
    assert result.desc == "invalid json"

    # Test valid json
    result = evaluator.evaluate_sample(test_instance, "{}")
    assert result.score == 0
    assert result.desc == "json does not match schema"

    valid_json_for_schema = {
        "title": "test",
        "description": "test",
        "termsOfServiceUrl": "https://test.com",
        "contact": "test@test.com",
        "license": "test",
        "licenseUrl": "https://test.com",
    }
    result = evaluator.evaluate_sample(test_instance, json.dumps(valid_json_for_schema))
    assert result.score == 1
    assert result.desc == "valid"

    missing_required_field_json = {
        "description": "test",
        "termsOfServiceUrl": "https://test.com",
        "contact": "test@test.com",
        "license": "test",
        "licenseUrl": "https://test.com",
    }

    result = evaluator.evaluate_sample(
        test_instance, json.dumps(missing_required_field_json)
    )
    assert result.score == 0
    assert result.desc == "json does not match schema"

    invalid_type_json = {
        "title": 1,
        "description": "test",
    }
    result = evaluator.evaluate_sample(test_instance, json.dumps(invalid_type_json))
    assert result.score == 0
    assert result.desc == "json does not match schema"

    invalid_format_json = {
        "title": "test",
        "description": "test",
        "termsOfServiceUrl": "not a url",
        "contact": "not an email",
        "license": "test",
        "licenseUrl": "not a url",
    }
    result = evaluator.evaluate_sample(test_instance, json.dumps(invalid_format_json))
    assert result.score == 0
    assert result.desc == "json does not match schema"


def test_prompt_formatter(test_instance):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    print(
        json.dumps(
            few_shots_messages_formatter(
                "Github_easy", TEST_SCHEMA, DEFAULT_SYSTEM_PROMPT
            ),
            indent=4,
        )
    )

    default_prompt_formatter(
        tokenizer,
        test_instance,
        use_chat_format=True,
    )

    with pytest.raises(NotImplementedError):
        default_prompt_formatter(
            tokenizer,
            test_instance,
            use_chat_format=False,
        )
