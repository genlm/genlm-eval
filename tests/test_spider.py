import pytest
import asyncio
from pathlib import Path
from genlm.control import direct_token_sampler, PromptedLLM

from genlm.eval.core import EvaluationResult, ModelOutput, ModelResponse, run_evaluation
from genlm.eval.domains.spider import (
    SpiderDataset,
    SpiderInstance,
    SpiderEvaluator,
    SYSTEM_PROMPT,
    SpiderTableColumnVerifier,
)


@pytest.fixture
def spider_data_dir():
    return Path(__file__).parent / "data/spider_data"


@pytest.fixture
def spider_dataset(spider_data_dir):
    return SpiderDataset.from_spider_dir(spider_data_dir, few_shot_example_ids=[0, 1])


@pytest.fixture
def spider_evaluator(spider_data_dir):
    return SpiderEvaluator(spider_data_dir)


def test_spider_data(spider_dataset):
    assert spider_dataset.schema is SpiderInstance
    for instance in spider_dataset:
        str(instance)
        assert isinstance(instance, SpiderInstance)


def test_spider_evaluator(spider_dataset, spider_evaluator):
    first_instance = next(iter(spider_dataset))

    assert spider_evaluator.evaluate_sample(
        first_instance, "SELECT count(*) FROM singer"
    ) == EvaluationResult(score=1.0, desc="valid", metadata={"level": "easy"})

    ensemble_result = spider_evaluator.evaluate_ensemble(
        first_instance,
        ModelOutput(
            responses=[
                ModelResponse(text="SELECT count(*) FROM singer", prob=0.5),
                ModelResponse(text="SELECT count(*) FROM singer", prob=0.5),
            ],
            runtime_seconds=0.1,
        ),
    )

    assert ensemble_result["weighted_accuracy"] == 1.0
    assert ensemble_result["runtime_seconds"] == 0.1

    ensemble_result = spider_evaluator.evaluate_ensemble(
        first_instance,
        ModelOutput(
            responses=[
                ModelResponse(text="SELECT count(*) FROM siner", prob=0.5),
                ModelResponse(text="SELECT count(*) FROM singer", prob=0.5),
            ],
            runtime_seconds=0.1,
        ),
    )

    assert ensemble_result["weighted_accuracy"] == 0.5
    assert ensemble_result["runtime_seconds"] == 0.1


def test_run_evaluation(spider_dataset, spider_evaluator):
    LLM = PromptedLLM.from_name("gpt2", backend="hf", eos_tokens=[b"\n", b"\n\n"])

    def sampler_factory(instance):
        prompt = (
            SYSTEM_PROMPT
            + "\n"
            + "\n".join("\n".join(x) for x in instance.few_shot_examples)
            + "\n"
            + instance.utterance
        )

        LLM.prompt_ids = LLM.model.tokenizer.encode(prompt)

        return direct_token_sampler(LLM)

    def critic_factory(instance):
        return SpiderTableColumnVerifier(
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
                ModelResponse(text=sequence, prob=prob)
                for sequence, prob in sequences.decoded_posterior.items()
            ],
            runtime_seconds=0.1,
        )

    n_replicates = 1

    result = asyncio.run(
        run_evaluation(
            dataset=spider_dataset,
            evaluator=spider_evaluator,
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
def potential(spider_dataset):
    first_instance = next(iter(spider_dataset))

    potential = SpiderTableColumnVerifier(
        tables=first_instance.tables,
        grammar=first_instance.lark_grammar,
        verbosity=2,
    )

    return potential


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
        "SELECT t2.Name FROM singer AS t2",
        "SELECT t3.Name FROM singer AS t3",
        "SELECT t4.Name FROM singer AS t4",
        "SELECT t5.Name FROM singer AS t5",
        "SELECT singer.Name, concert.Theme FROM singer, concert",
        "SELECT Name FROM singer WHERE Age > 25",
        "SELECT Name FROM singer GROUP BY Country",
        "SELECT Name FROM singer ORDER BY Age DESC",
        "SELECT t1.Name, t2.concert_ID FROM singer t1 JOIN concert t2 ON t1.Name = t2.concert_name",
        "SELECT * FROM (SELECT Name FROM singer) WHERE (SELECT COUNT(*) FROM concert)",
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
        "SELECT Name FROM singer GROUP BY NonExistentColumn",
        "SELECT t1.Name, t2.concert_ID FROM singer t1 JOIN concert t2 ON t1.Name = t2.Name",
    ]
    for query in invalid_queries:
        result = await potential.complete(query.encode())
        assert result == float("-inf"), f"Failed for query: {query}"


def test_extract_latest_subquery():
    queries = [
        (
            "SELECT * FROM (SELECT Name FROM singer) WHERE (SELECT COUNT(*) FROM concert)",
            "SELECT COUNT(*) FROM concert",
        ),
        (
            "SELECT * FROM singer WHERE Singer_ID IN (SELECT Singer_ID FROM singer_in_concert)",
            "SELECT Singer_ID FROM singer_in_concert",
        ),
        ("SELECT * FROM (SELECT Name FROM singer) WHERE", "SELECT Name FROM singer"),
    ]
    for query, expected in queries:
        result = SpiderTableColumnVerifier._extract_latest_subquery(query)
        assert result == expected


def test_strip_query_at_boundary():
    test_cases = [
        ("SELECT Name FROM singer WHERE", "SELECT Name FROM singer"),
        ("SELECT Name FROM singer GROUP BY", "SELECT Name FROM singer"),
        ("SELECT Name FROM singer ORDER", "SELECT Name FROM singer"),
        ("SELECT Name FROM singer", None),
    ]
    for query, expected in test_cases:
        result = SpiderTableColumnVerifier._strip_query_at_boundary(query)
        assert result == expected


@pytest.mark.asyncio
async def test_invalid_utf8_input(potential):
    invalid_utf8 = b"\x80\x80\x80"
    assert await potential.prefix(invalid_utf8) == 0
    assert await potential.complete(invalid_utf8) == float("-inf")
