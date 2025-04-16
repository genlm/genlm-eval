import os
import json
from genlm.eval.core.model import ModelOutput


async def run_evaluation(
    dataset,
    model,
    evaluator,
    output_dir=None,
    n_replicates=1,
    overwrite_results=False,
    overwrite_output=False,
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
        overwrite_output (bool, optional): Whether to overwrite existing output. Defaults to False.
        verbosity (int, optional): The verbosity of the evaluation. Defaults to 0, which is silent.

    Returns:
        Dict[str, Any]: Aggregated evaluation results.
    """
    all_results = []
    all_instance_results = []
    all_instance_outputs = []

    if overwrite_output and not overwrite_results:
        raise ValueError("Cannot overwrite output without overwriting results")

    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

                if not overwrite_output and os.path.exists(instance_output_path):
                    try:
                        with open(instance_output_path, "r") as f:
                            output = ModelOutput.model_validate_json(f.read())
                    except json.JSONDecodeError:
                        pass

                if not overwrite_results and os.path.exists(instance_results_path):
                    try:
                        with open(instance_results_path, "r") as f:
                            result = json.load(f)
                    except json.JSONDecodeError:
                        pass
            else:
                record_path = None
                instance_output_path = None
                instance_results_path = None

            if output is None:
                output = await model.generate(instance, record_path)

                if instance_output_path is not None:
                    with open(instance_output_path, "w") as f:
                        f.write(output.model_dump_json(indent=4))

            if result is None or overwrite_results:
                result = evaluator.evaluate_ensemble(instance, output)

                if instance_results_path is not None:
                    with open(instance_results_path, "w") as f:
                        f.write(result.model_dump_json(indent=4))

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
