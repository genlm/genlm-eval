import pytest
import asyncio
from pathlib import Path

from genlm.control import direct_token_sampler

from genlm.eval.domains.spider.model import SpiderModel
from genlm.eval.domains.spider.dataset import SpiderDataset, SpiderInstance
from genlm.eval.domains.spider.evaluator import SpiderEvaluator

from genlm.eval.core import EvaluationResult, ModelOutput, ModelResponse, run_evaluation


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

    assert spider_evaluator.evaluate_response(
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


def test_spider_model_init():
    model = SpiderModel(
        model_name="placeholder",  # should be ignored so long as we don't access .llm
        max_tokens=100,
        n_particles=2,
        ess_threshold=0.5,
    )

    assert model.max_tokens == 100
    assert model.n_particles == 2
    assert model.ess_threshold == 0.5
    assert model.bool_cfg("concert_singer") is not None


def test_run_evaluation(spider_dataset, spider_evaluator):
    class TestSpiderModel(SpiderModel):
        def make_sampler(self, instance):
            return direct_token_sampler(self.llm)

        def make_critic(self, instance):
            return

    n_particles = 2

    model = TestSpiderModel(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        max_tokens=100,
        n_particles=n_particles,
        ess_threshold=0.5,
        lm_args={"backend": "hf"},
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

    assert result["average_weighted_accuracy"] > 0.0
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
