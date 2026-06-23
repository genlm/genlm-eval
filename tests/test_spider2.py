import asyncio
import json
from pathlib import Path

import pandas as pd
import pytest
from genlm.control import direct_token_sampler, PromptedLLM

from genlm.eval.core import EvaluationResult, ModelOutput, ModelResponse, run_evaluation
from genlm.eval.domains.spider2 import (
    backend_for_instance,
    Spider2Dataset,
    Spider2Evaluator,
    Spider2Instance,
    Spider2TableColumnVerifier,
    default_prompt_formatter,
)
from genlm.eval.domains.spider2.spider2_eval import evaluator as evaluator_mod


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


###################
# Backend dispatch #
###################


def test_backend_for_instance():
    assert backend_for_instance("local_concert_001") == "sqlite"
    assert backend_for_instance("sf_bq001") == "snowflake"
    assert backend_for_instance("sf123") == "snowflake"
    assert backend_for_instance("bq001") == "bigquery"
    assert backend_for_instance("ga004") == "bigquery"


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def _row(instance_id):
    return {
        "instance_id": instance_id,
        "db": "d",
        "question": "q",
        "external_knowledge": None,
    }


def test_default_filter_includes_snowflake_skips_bigquery(tmp_path):
    _write_jsonl(
        tmp_path / "spider2-lite.jsonl",
        [_row("local_x"), _row("sf_bq_x"), _row("bq_x")],
    )

    ds = Spider2Dataset.from_spider2_dir(tmp_path)
    assert {inst.spider2_instance_id for inst in ds} == {"local_x", "sf_bq_x"}

    ds_sqlite = Spider2Dataset.from_spider2_dir(tmp_path, enabled_backends=("sqlite",))
    assert {inst.spider2_instance_id for inst in ds_sqlite} == {"local_x"}


def test_snowflake_dispatch(tmp_path, monkeypatch):
    cred = tmp_path / "snowflake_credential.json"
    cred.write_text(
        json.dumps({"account": "a", "username": "u", "password": "p"}),
        encoding="utf-8",
    )

    def fake_execute_snowflake(credential, query, timeout=None):
        values = [1, 2, 3] if "good" in query else [9]
        return pd.DataFrame({"c": values})

    monkeypatch.setattr(evaluator_mod, "execute_snowflake", fake_execute_snowflake)

    evaluator = Spider2Evaluator(tmp_path, snowflake_credential_path=cred).evaluator

    ok, reason, _ = evaluator.evaluate("SELECT good", "SELECT good", "d", "sf_bq001")
    assert ok and reason is None

    ok, reason, _ = evaluator.evaluate("SELECT bad", "SELECT good", "d", "sf_bq001")
    assert not ok and reason == "mismatch"


def test_snowflake_missing_credentials(tmp_path):
    evaluator = Spider2Evaluator(tmp_path).evaluator
    ok, reason, _ = evaluator.evaluate("SELECT 1", "SELECT 1", "d", "sf_bq001")
    assert not ok and reason == "missing snowflake credentials"


def test_bigquery_missing_project(tmp_path):
    # BigQuery is opt-in: with no project configured it is not executed.
    evaluator = Spider2Evaluator(tmp_path).evaluator
    ok, reason, _ = evaluator.evaluate("SELECT 1", "SELECT 1", "d", "bq001")
    assert not ok and reason == "missing bigquery project"


def test_bigquery_dispatch(tmp_path, monkeypatch):
    def fake_execute_bigquery(client, query, timeout=None):
        values = [1, 2, 3] if "good" in query else [9]
        return pd.DataFrame({"c": values})

    monkeypatch.setattr(evaluator_mod, "execute_bigquery", fake_execute_bigquery)

    evaluator = Spider2Evaluator(tmp_path, bigquery_project="proj").evaluator
    # Stub the client builder so no Google library import is needed.
    monkeypatch.setattr(evaluator, "_get_bigquery_client", lambda: object())

    ok, reason, _ = evaluator.evaluate("SELECT good", "SELECT good", "d", "bq001")
    assert ok and reason is None

    ok, reason, _ = evaluator.evaluate("SELECT bad", "SELECT good", "d", "bq001")
    assert not ok and reason == "mismatch"


def test_evaluate_defaults_to_sqlite(tmp_path):
    # instance_id=None -> sqlite backend; missing db surfaces the sqlite reason.
    evaluator = Spider2Evaluator(tmp_path).evaluator
    ok, reason, _ = evaluator.evaluate("SELECT 1", "SELECT 1", "nodb", instance_id=None)
    assert not ok and reason == "missing sqlite database `nodb`"


########################
# execute_snowflake    #
########################


def test_execute_snowflake_connection_and_dataframe(monkeypatch):
    """Exercise the real ``execute_snowflake`` body with a fake connector module."""
    import sys
    import types

    captured = {}

    class FakeCursor:
        description = [("a",), ("b",)]

        def execute(self, query):
            captured["query"] = query

        def fetchall(self):
            return [(1, "x"), (2, "y")]

    class FakeConn:
        def cursor(self):
            return FakeCursor()

        def close(self):
            captured["closed"] = True

    def fake_connect(**kwargs):
        captured["connect"] = kwargs
        return FakeConn()

    fake_pkg = types.ModuleType("snowflake")
    fake_mod = types.ModuleType("snowflake.connector")
    fake_mod.connect = fake_connect
    fake_pkg.connector = fake_mod
    monkeypatch.setitem(sys.modules, "snowflake", fake_pkg)
    monkeypatch.setitem(sys.modules, "snowflake.connector", fake_mod)

    credential = {
        "account": "ACC",
        "username": "USER",
        "password": "TOK",
        "role": "PARTICIPANT",
        "warehouse": "WH",
    }
    df = evaluator_mod.execute_snowflake(credential, "SELECT 1", timeout=30)

    # username -> user; role/warehouse forwarded; timeout -> login_timeout.
    assert captured["connect"] == {
        "account": "ACC",
        "user": "USER",
        "password": "TOK",
        "role": "PARTICIPANT",
        "warehouse": "WH",
        "login_timeout": 30,
    }
    assert captured["query"] == "SELECT 1"
    assert captured["closed"] is True
    assert list(df.columns) == ["a", "b"]
    assert df["a"].tolist() == [1, 2]


########################
# Result comparison    #
########################


def test_compare_identical_and_numeric_tolerance():
    gold = pd.DataFrame({"x": [1.0, 2.0]})
    assert evaluator_mod.compare_pandas_table(gold.copy(), gold)
    assert evaluator_mod.compare_pandas_table(pd.DataFrame({"x": [1.005, 2.0]}), gold)
    assert not evaluator_mod.compare_pandas_table(pd.DataFrame({"x": [1.5, 2.0]}), gold)


def test_compare_ignore_order():
    gold = pd.DataFrame({"x": [1, 2, 3]})
    pred = pd.DataFrame({"x": [3, 1, 2]})
    assert evaluator_mod.compare_pandas_table(pred, gold, ignore_order=True)
    assert not evaluator_mod.compare_pandas_table(pred, gold, ignore_order=False)


def test_compare_extra_and_missing_columns():
    gold = pd.DataFrame({"x": [1, 2]})
    # An extra predicted column is fine as long as every gold column is matched.
    assert evaluator_mod.compare_pandas_table(
        pd.DataFrame({"x": [1, 2], "y": [9, 9]}), gold
    )
    # Fewer predicted columns than gold -> fail.
    gold2 = pd.DataFrame({"x": [1, 2], "z": [3, 4]})
    assert not evaluator_mod.compare_pandas_table(pd.DataFrame({"x": [1, 2]}), gold2)


def test_compare_condition_cols():
    gold = pd.DataFrame({"keep": [1, 2], "ignore": [5, 6]})
    pred = pd.DataFrame({"keep": [1, 2]})
    # Only gold column 0 is required when condition_cols is set.
    assert evaluator_mod.compare_pandas_table(pred, gold, condition_cols=[0])
    # Without it, gold's second column is unmatched -> fail.
    assert not evaluator_mod.compare_pandas_table(pred, gold)


########################
# Gold / config loaders #
########################


def test_load_gold_results_matches_variants(tmp_path):
    (tmp_path / "sf_bq001.csv").write_text("a\n1\n", encoding="utf-8")
    (tmp_path / "sf_bq001_a.csv").write_text("a\n2\n", encoding="utf-8")
    (tmp_path / "sf_bq002.csv").write_text("a\n3\n", encoding="utf-8")

    results = evaluator_mod.load_gold_results(tmp_path, "sf_bq001")
    assert len(results) == 2
    assert sorted(int(r["a"].iloc[0]) for r in results) == [1, 2]
    # A missing directory yields no results rather than raising.
    assert evaluator_mod.load_gold_results(tmp_path / "nope", "sf_bq001") == []


def test_load_eval_configs(tmp_path):
    path = tmp_path / "cfg.jsonl"
    path.write_text(
        json.dumps(
            {"instance_id": "sf_bq001", "condition_cols": [0, 2], "ignore_order": False}
        )
        + "\n\n",  # trailing blank line should be skipped
        encoding="utf-8",
    )
    cfgs = evaluator_mod.load_eval_configs(path)
    assert cfgs["sf_bq001"].condition_cols == [0, 2]
    assert cfgs["sf_bq001"].ignore_order is False


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
