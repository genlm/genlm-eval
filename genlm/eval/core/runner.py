import os
import json
import pydantic
from genlm.eval.core.model import ModelOutput


def _load_cached_output(path):
    """Try to load cached output, return None if invalid."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return ModelOutput.model_validate_json(f.read())
    except (json.JSONDecodeError, pydantic.ValidationError):
        return None


def _load_cached_results(path):
    """Try to load cached results, return None if invalid."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def _save_output(output, path):
    """Save model output to file."""
    with open(path, "w") as f:
        f.write(output.model_dump_json(indent=4))


def _save_results(results, path):
    """Save evaluation results to file."""
    with open(path, "w") as f:
        json.dump(results, f, indent=4)


async def run_evaluation(
    dataset,
    model,
    evaluator,
    output_dir=None,
    n_replicates=1,
    overwrite_results=False,
    overwrite_outputs=False,
    verbosity=0,
):
    """Run evaluation on a dataset using the provided model and evaluator.

    Args:
        dataset (Dataset): The dataset to evaluate on.
        model_adaptor (ModelAdaptor): The model adaptor to use for generation.
        evaluator (Evaluator): The evaluator to use for prompt generation and evaluation.
        output_dir (str, optional): The directory to save the results. Defaults to None, in which case results are not saved.
        n_replicates (int, optional): Number of times to replicate the evaluation. Defaults to 1.
        overwrite_results (bool, optional): Whether to overwrite existing evaluation results. Defaults to False.
        overwrite_outputs (bool, optional): Whether to overwrite existing output. Defaults to False.
        verbosity (int, optional): The verbosity of the evaluation. Defaults to 0, which is silent.

    Returns:
        Dict[str, Any]: Aggregated evaluation results.
    """
    all_results = []
    all_instance_results = []
    all_instance_outputs = []

    if overwrite_outputs and not overwrite_results:
        raise ValueError(
            "Cannot overwrite outputs without overwriting results. (Hint: set overwrite_results=True)"
        )

    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)  # pragma: no cover

    for instance in dataset:
        instance_results = []
        instance_outputs = []
        instance_id = instance.instance_id

        for i in range(n_replicates):
            output = None
            result = None
            if output_dir is not None:
                record_path = os.path.join(output_dir, f"{instance_id}-{i}-record.json")
                instance_output_path = os.path.join(
                    output_dir, f"{instance_id}-{i}-output.json"
                )
                instance_results_path = os.path.join(
                    output_dir, f"{instance_id}-{i}-results.json"
                )

                # Try loading cached files if not overwriting
                if not overwrite_outputs:
                    output = _load_cached_output(instance_output_path)
                if not overwrite_results:
                    result = _load_cached_results(instance_results_path)
            else:
                record_path = None
                instance_output_path = None
                instance_results_path = None

            # Generate new output if needed
            wrote_output = False
            if output is None:
                output = await model.generate(instance, record_path)
                if instance_output_path is not None:
                    wrote_output = True
                    _save_output(output, instance_output_path)

            # Evaluate if we need new results (no results, overwriting results, or wrote new output)
            if result is None or overwrite_results or wrote_output:
                result = evaluator.evaluate_ensemble(instance, output)
                if instance_results_path is not None:
                    _save_results(result, instance_results_path)

            instance_results.append(result)
            instance_outputs.append(output)

        avg_instance_result = {
            "weighted_accuracy": sum(r["weighted_accuracy"] for r in instance_results)
            / n_replicates,
        }
        all_results.append(avg_instance_result)
        all_instance_results.append(instance_results)
        all_instance_outputs.append(instance_outputs)

        if verbosity > 0:
            print(f"Instance {instance}")
            print(
                f"Mean weighted accuracy (instance): {avg_instance_result['weighted_accuracy']}"
            )
            print(
                f"Mean weighted accuracy (total): {sum(r['weighted_accuracy'] for r in all_results) / len(all_results)}"
            )
            print()

    return {
        "average_weighted_accuracy": sum(r["weighted_accuracy"] for r in all_results)
        / len(all_results),
        "n_instances": len(all_results),
        "all_instance_results": all_instance_results,
        "all_instance_outputs": all_instance_outputs,
    }
