import os
import pytest
import asyncio
import tempfile
from genlm.control import direct_token_sampler, PromptedLLM

from genlm.eval.core import ModelOutput, ModelResponse, run_evaluation
from genlm.eval.domains.pattern_matching import (
    default_prompt_formatter,
    PatternPotential,
    PatternMatchingDataset,
    PatternMatchingInstance,
    PatternMatchingEvaluator,
)


@pytest.fixture
def dataset():
    return PatternMatchingDataset(["a|d", "b"])


@pytest.fixture
def evaluator():
    return PatternMatchingEvaluator()


def test_data(dataset):
    assert dataset.schema is PatternMatchingInstance
    for instance in dataset:
        assert isinstance(instance, PatternMatchingInstance)
        repr(instance)


def test_from_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "patterns.csv")
        with open(csv_path, "w") as f:
            f.write("pattern\n")
            f.write("a|d\n")
            f.write("b\n")
        dataset = PatternMatchingDataset.from_csv(csv_path, "pattern")
        assert len(dataset) == 2
        assert dataset.patterns == ["a|d", "b"]


@pytest.mark.asyncio
async def test_potential():
    potential = PatternPotential("a|d")
    assert await potential.prefix(b"") == 0.0
    assert await potential.complete(b"") == float("-inf")
    assert await potential.prefix(b"a") == 0.0
    assert await potential.complete(b"a") == 0.0
    assert await potential.prefix(b"d") == 0.0

    potential = PatternPotential("abc")
    assert await potential.prefix(b"") == 0.0
    assert await potential.complete(b"") == float("-inf")
    assert await potential.prefix(b"a") == 0.0
    assert await potential.complete(b"b") == float("-inf")
    assert await potential.prefix(b"c") == float("-inf")
    assert await potential.complete(b"abc") == 0.0


def test_evaluator(dataset, evaluator):
    first_instance = next(iter(dataset))

    result = evaluator.evaluate_sample(first_instance, "a")
    assert result.score == 1
    assert result.desc == "valid"

    result = evaluator.evaluate_sample(first_instance, "b")
    assert result.score == 0
    assert result.desc == "invalid"

    ensemble_result = evaluator.evaluate_ensemble(
        first_instance,
        ModelOutput(
            responses=[
                ModelResponse(response="a", weight=0.5),
                ModelResponse(response="b", weight=0.5),
            ],
            runtime_seconds=0.1,
        ),
    )

    assert ensemble_result["weighted_accuracy"] == 0.5
    assert ensemble_result["runtime_seconds"] == 0.1


def test_run_evaluation(dataset, evaluator):
    LLM = PromptedLLM.from_name("gpt2", backend="hf", eos_tokens=[b"\n", b"\n\n"])

    def sampler_factory(instance):
        LLM.prompt_ids = default_prompt_formatter(
            LLM.model.tokenizer, instance, use_chat_format=False
        )
        return direct_token_sampler(LLM)

    def critic_factory(instance):
        return PatternPotential(instance.pattern).coerce(LLM, f=b"".join)

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

    result = asyncio.run(
        run_evaluation(
            dataset=dataset,
            evaluator=evaluator,
            model=model,
            n_replicates=1,
        )
    )

    assert result["n_instances"] == 2
