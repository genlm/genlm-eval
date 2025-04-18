import pytest
from unittest.mock import AsyncMock, MagicMock
import os
import json
import tempfile
from genlm.eval.core.runner import run_evaluation
from genlm.eval.core.model import ModelOutput, ModelResponse


@pytest.fixture
def mock_dataset():
    dataset = MagicMock()
    instance = MagicMock()
    instance.instance_id = "test_instance"
    dataset.__iter__.return_value = [instance]
    return dataset


@pytest.fixture
def mock_model():
    model = AsyncMock()
    model.generate.return_value = ModelOutput(
        responses=[ModelResponse(text="test output", prob=0.8, metadata={"time": 1.0})],
        runtime_seconds=1.0,
    )
    return model


@pytest.fixture
def mock_evaluator():
    evaluator = MagicMock()
    evaluator.evaluate_ensemble.return_value = {"weighted_accuracy": 0.8}
    return evaluator


@pytest.mark.asyncio
async def test_basic_evaluation(mock_dataset, mock_model, mock_evaluator):
    results = await run_evaluation(
        dataset=mock_dataset,
        model=mock_model,
        evaluator=mock_evaluator,
    )

    assert "average_weighted_accuracy" in results
    assert results["average_weighted_accuracy"] == 0.8
    assert results["n_instances"] == 1
    assert len(results["all_instance_results"]) == 1
    assert len(results["all_instance_outputs"]) == 1


@pytest.mark.asyncio
async def test_evaluation_with_replicates(mock_dataset, mock_model, mock_evaluator):
    results = await run_evaluation(
        dataset=mock_dataset,
        model=mock_model,
        evaluator=mock_evaluator,
        n_replicates=3,
        verbosity=1,
    )

    assert results["n_instances"] == 1
    assert len(results["all_instance_results"][0]) == 3
    assert len(results["all_instance_outputs"][0]) == 3


@pytest.mark.asyncio
async def test_evaluation_with_output_dir(mock_dataset, mock_model, mock_evaluator):
    with tempfile.TemporaryDirectory() as tmpdir:
        await run_evaluation(
            dataset=mock_dataset,
            model=mock_model,
            evaluator=mock_evaluator,
            output_dir=tmpdir,
        )

        # Check if files were created
        instance_files = os.listdir(tmpdir)
        assert len(instance_files) == 2  # output.json, results.json
        assert any(f.endswith("-output.json") for f in instance_files)
        assert any(f.endswith("-results.json") for f in instance_files)


@pytest.mark.asyncio
async def test_evaluation_with_existing_output(
    mock_dataset, mock_model, mock_evaluator
):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create existing output file
        instance_output = ModelOutput(
            responses=[
                ModelResponse(text="existing output", prob=0.8, metadata={"time": 2.0})
            ],
            runtime_seconds=2.0,
        )
        output_path = os.path.join(tmpdir, "test_instance-0-output.json")
        with open(output_path, "w") as f:
            f.write(instance_output.model_dump_json(indent=4))

        await run_evaluation(
            dataset=mock_dataset,
            model=mock_model,
            evaluator=mock_evaluator,
            output_dir=tmpdir,
            overwrite_outputs=False,
        )

        mock_model.generate.assert_not_called()


@pytest.mark.asyncio
async def test_invalid_overwrite_configuration(
    mock_dataset, mock_model, mock_evaluator
):
    with pytest.raises(
        ValueError, match="Cannot overwrite output without overwriting results"
    ):
        await run_evaluation(
            dataset=mock_dataset,
            model=mock_model,
            evaluator=mock_evaluator,
            overwrite_outputs=True,
            overwrite_results=False,
        )


@pytest.mark.asyncio
async def test_evaluation_with_existing_results_only(
    mock_dataset, mock_model, mock_evaluator
):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create existing results file only
        existing_results = {"weighted_accuracy": 0.9}
        results_path = os.path.join(tmpdir, "test_instance-0-results.json")
        with open(results_path, "w") as f:
            json.dump(existing_results, f, indent=4)

        await run_evaluation(
            dataset=mock_dataset,
            model=mock_model,
            evaluator=mock_evaluator,
            output_dir=tmpdir,
            overwrite_outputs=False,
            overwrite_results=True,
        )

        # Model.generate should have been called since there's no output file
        mock_model.generate.assert_called_once()
        # If we have to write output, the evaluator should have been called
        mock_evaluator.evaluate_ensemble.assert_called_once()


@pytest.mark.asyncio
async def test_evaluation_with_invalid_json_files(
    mock_dataset, mock_model, mock_evaluator
):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create invalid output file
        output_path = os.path.join(tmpdir, "test_instance-0-output.json")
        with open(output_path, "w") as f:
            f.write("invalid json{")

        # Create invalid results file
        results_path = os.path.join(tmpdir, "test_instance-0-results.json")
        with open(results_path, "w") as f:
            f.write("also invalid json{")

        await run_evaluation(
            dataset=mock_dataset,
            model=mock_model,
            evaluator=mock_evaluator,
            output_dir=tmpdir,
            overwrite_outputs=False,
            overwrite_results=False,
        )

        # Both generate and evaluate should be called since files were invalid
        mock_model.generate.assert_called_once()
        mock_evaluator.evaluate_ensemble.assert_called_once()

        # Verify new valid files were written
        with open(output_path) as f:
            # Should not raise JSONDecodeError
            json.load(f)
        with open(results_path) as f:
            # Should not raise JSONDecodeError
            json.load(f)


@pytest.mark.asyncio
async def test_evaluation_with_invalid_output_valid_results(
    mock_dataset, mock_model, mock_evaluator
):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create invalid output file
        output_path = os.path.join(tmpdir, "test_instance-0-output.json")
        with open(output_path, "w") as f:
            f.write("invalid json{")

        # Create valid results file
        existing_results = {"weighted_accuracy": 0.9}
        results_path = os.path.join(tmpdir, "test_instance-0-results.json")
        with open(results_path, "w") as f:
            json.dump(existing_results, f, indent=4)

        await run_evaluation(
            dataset=mock_dataset,
            model=mock_model,
            evaluator=mock_evaluator,
            output_dir=tmpdir,
            overwrite_outputs=False,
            overwrite_results=False,
        )

        # Generate should be called since output file was invalid
        mock_model.generate.assert_called_once()
        # Evaluator should be called since output file was overwritten
        mock_evaluator.evaluate_ensemble.assert_called_once()
