import asyncio
from pathlib import Path

import pytest
from genlm.control import direct_token_sampler, PromptedLLM

from genlm.eval.core import EvaluationResult, ModelOutput, ModelResponse, run_evaluation
from genlm.eval.domains.spider2 import (
    Spider2Dataset,
    Spider2Evaluator,
    Spider2Instance,
    Spider2TableColumnVerifier,
    default_prompt_formatter,
)


@pytest.fixture
def spider2_dir():
    return Path(__file__).parent.parent / "assets" / "spider2"


@pytest.fixture
def spider2_data_dir(spider2_dir):
    return spider2_dir / "spider2_sample"


@pytest.fixture
def spider2_grammars(spider2_data_dir):
    return spider2_data_dir / "grammars.json"


@pytest.fixture
def spider2_dataset(spider2_data_dir, spider2_grammars):
    return Spider2Dataset.from_spider2_dir(
        spider2_data_dir,
        grammar_json_path=spider2_grammars,
        few_shot_example_ids=[0, 1],
    )


@pytest.fixture
def spider2_evaluator(spider2_data_dir):
    return Spider2Evaluator(spider2_data_dir)


def test_spider2_data(spider2_dataset):
    assert spider2_dataset.schema is Spider2Instance
    for instance in spider2_dataset:
        str(instance)
        assert isinstance(instance, Spider2Instance)
        assert instance.spider2_instance_id.startswith("local")
        assert instance.schema_str
        # tables should round-trip from the parsed DDL
        assert {t.name for t in instance.tables} >= {"singer"}


def test_spider2_evaluator(spider2_dataset, spider2_evaluator):
    first_instance = next(iter(spider2_dataset))

    assert spider2_evaluator.evaluate_sample(
        first_instance, "SELECT count(*) FROM singer;"
    ) == EvaluationResult(score=1.0, desc="valid", metadata={})

    ensemble_result = spider2_evaluator.evaluate_ensemble(
        first_instance,
        ModelOutput(
            responses=[
                ModelResponse(response="SELECT count(*) FROM singer;", weight=0.5),
                ModelResponse(response="SELECT count(*) FROM singer;", weight=0.5),
            ],
            runtime_seconds=0.1,
        ),
    )
    assert ensemble_result["weighted_accuracy"] == 1.0
    assert ensemble_result["runtime_seconds"] == 0.1

    ensemble_result = spider2_evaluator.evaluate_ensemble(
        first_instance,
        ModelOutput(
            responses=[
                ModelResponse(response="SELECT count(*) FROM siner;", weight=0.5),
                ModelResponse(response="SELECT count(*) FROM singer;", weight=0.5),
            ],
            runtime_seconds=0.1,
        ),
    )
    assert ensemble_result["weighted_accuracy"] == 0.5
    assert ensemble_result["runtime_seconds"] == 0.1


def test_run_evaluation(spider2_dataset, spider2_evaluator):
    LLM = PromptedLLM.from_name("gpt2", backend="hf", eos_tokens=[b"\n", b"\n\n"])

    def sampler_factory(instance):
        LLM.prompt_ids = default_prompt_formatter(
            LLM.model.tokenizer, instance, use_chat_format=False
        )
        return direct_token_sampler(LLM)

    def critic_factory(instance):
        return Spider2TableColumnVerifier(
            tables=instance.tables,
            grammar=instance.lark_grammar,
        ).coerce(LLM, f=b"".join)

    async def model(instance, output_dir, replicate):
        assert replicate == 0
        sequences = await sampler_factory(instance).smc(
            critic=critic_factory(instance),
            n_particles=2,
            ess_threshold=0.5,
            max_tokens=100,
        )
        return ModelOutput(
            responses=[
                ModelResponse(response=sequence, weight=prob)
                for sequence, prob in sequences.decoded_posterior.items()
            ],
            runtime_seconds=0.1,
        )

    n_replicates = 1
    result = asyncio.run(
        run_evaluation(
            dataset=spider2_dataset,
            evaluator=spider2_evaluator,
            model=model,
            n_replicates=n_replicates,
        )
    )

    assert result["n_instances"] == 2
    for instance_result in result["all_instance_results"]:
        assert len(instance_result) == n_replicates
        for r in instance_result:
            assert "weighted_accuracy" in r
            assert "runtime_seconds" in r

    for instance_output in result["all_instance_outputs"]:
        assert len(instance_output) == n_replicates
        for response in instance_output:
            assert isinstance(response, ModelOutput)
            for r in response.responses:
                assert isinstance(r, ModelResponse)


@pytest.fixture
def potential(spider2_dataset):
    first_instance = next(iter(spider2_dataset))
    return Spider2TableColumnVerifier(
        tables=first_instance.tables,
        grammar=first_instance.lark_grammar,
        verbosity=2,
    )


@pytest.mark.asyncio
async def test_prefix_valid_queries(potential):
    valid_queries = [
        "SELECT Name FROM",
        "SELECT Name FROM singer WHERE",
        "SELECT t1.Name FROM singer AS t1 WHERE",
        "SELECT COUNT(*) FROM singer WHERE",
        "SELECT singer.Name, concert.Theme FROM singer, concert WHERE",
        "SELECT DISTINCT Name FROM singer WHERE",
        "SELECT * FROM (SELECT Name FROM singer WHERE",
    ]
    for query in valid_queries:
        result = await potential.prefix(query.encode())
        assert result == 0, f"Failed for query: {query}"


@pytest.mark.asyncio
async def test_prefix_invalid_queries(potential):
    invalid_queries = [
        "SELECT Stadium_ID FROM singer WHERE",
        "SELECT t1.concert_ID FROM stadium AS t1 WHERE",
        "SELECT t2.Name FROM singer AS t1 WHERE",
    ]
    for query in invalid_queries:
        result = await potential.prefix(query.encode())
        assert result == float("-inf"), f"Failed for query: {query}"


@pytest.mark.asyncio
async def test_complete_valid_queries(potential):
    valid_queries = [
        "SELECT Name FROM singer",
        "SELECT COUNT(*) FROM singer",
        "SELECT t1.Name FROM singer AS t1",
        "SELECT singer.Name, concert.Theme FROM singer, concert",
        "SELECT Name FROM singer WHERE Age > 25",
        "SELECT Name FROM singer GROUP BY Country",
        "SELECT Name FROM singer ORDER BY Age DESC",
    ]
    for query in valid_queries:
        result = await potential.complete(query.encode())
        assert result == 0, f"Failed for query: {query}"


@pytest.mark.asyncio
async def test_complete_invalid_queries(potential):
    invalid_queries = [
        "SELECT NonExistentColumn FROM singer",
        "SELECT Name FROM nonexistent_table",
        "SELECT * FROM singer WHERE NonExistentColumn = 5",
    ]
    for query in invalid_queries:
        result = await potential.complete(query.encode())
        assert result == float("-inf"), f"Failed for query: {query}"
