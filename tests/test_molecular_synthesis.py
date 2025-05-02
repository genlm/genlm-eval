import os
import pytest
import asyncio
import tempfile
from genlm.control import direct_token_sampler, PromptedLLM

from genlm.eval.core import ModelOutput, ModelResponse, run_evaluation
from genlm.eval.domains.molecular_synthesis import (
    default_prompt_formatter,
    PartialSMILES,
    MolecularSynthesisDataset,
    MolecularSynthesisInstance,
    MolecularSynthesisEvaluator,
)


@pytest.fixture
def mol_dataset():
    return MolecularSynthesisDataset(
        [
            [
                "BrC1=C2C3C4C3N(CC4C#C)C2=NC(=O)S1",
                "BrC1=C2C3C4CC4C(C3C=O)C2=NNS1(=O)=O",
            ],
            [
                "BrC1=C2C3C4C3N(CC4C#C)C2=NC(=O)S1",
                "BrC1=C2C3C4CC4C(C3C=O)C2=NNS1(=O)=O",
            ],
        ]
    )


@pytest.fixture
def mol_evaluator():
    return MolecularSynthesisEvaluator()


def test_from_smiles():
    with tempfile.TemporaryDirectory() as tmpdir:
        smiles_path = os.path.join(tmpdir, "test_molecules.smi")
        with open(smiles_path, "w") as f:
            f.write("BrC1=C2C3C4C3N(CC4C#C)C2=NC(=O)S1\n")
            f.write("BrC1=C2C3C4CC4C(C3C=O)C2=NNS1(=O)=O\n")
        dataset = MolecularSynthesisDataset.from_smiles(
            smiles_path, n_molecules=2, n_instances=2
        )
        assert len(dataset) == 2


def test_mol_data(mol_dataset):
    assert mol_dataset.schema is MolecularSynthesisInstance
    for instance in mol_dataset:
        assert isinstance(instance, MolecularSynthesisInstance)


def test_mol_evaluator(mol_dataset, mol_evaluator):
    first_instance = next(iter(mol_dataset))
    valid_smiles = "BrC1=C2C3C4C3N(CC4C#C)C2=NC(=O)S1"

    # Test with a valid SMILES string
    result = mol_evaluator.evaluate_sample(first_instance, valid_smiles)
    assert result.score > 0
    assert result.desc == "valid"

    ensemble_result = mol_evaluator.evaluate_ensemble(
        first_instance,
        ModelOutput(
            responses=[
                ModelResponse(response=valid_smiles, weight=0.5),
                ModelResponse(response=valid_smiles, weight=0.5),
            ],
            runtime_seconds=0.1,
        ),
    )

    assert ensemble_result["weighted_accuracy"] > 0.0
    assert ensemble_result["runtime_seconds"] == 0.1

    # Test with invalid SMILES
    ensemble_result = mol_evaluator.evaluate_ensemble(
        first_instance,
        ModelOutput(
            responses=[
                ModelResponse(response="invalid_smiles", weight=0.5),
                ModelResponse(response=valid_smiles, weight=0.5),
            ],
            runtime_seconds=0.1,
        ),
    )

    assert 0.0 < ensemble_result["weighted_accuracy"] <= 0.5
    assert ensemble_result["runtime_seconds"] == 0.1


@pytest.mark.asyncio
async def test_potential():
    potential = PartialSMILES()

    assert await potential.prefix(b"") == 0.0
    assert await potential.complete(b"") == float("-inf")

    assert await potential.prefix(b"BrC1=C2C3C4C3N(CC4C#C)C2=NC(=O)") == 0.0
    assert await potential.complete(b"BrC1=C2C3C4C3N(CC4C#C)C2=NC(=O)S1") == 0.0

    assert await potential.prefix(b" BrC1=C2C3C4C3N(CC4C#C)C2=NC(=O)") == 0.0
    assert await potential.complete(b" BrC1=C2C3C4C3N(CC4C#C)C2=NC(=O)S1") == 0.0

    assert await potential.prefix(b"23r23") == float("-inf")
    assert await potential.complete(b"2323r2") == float("-inf")


def test_run_evaluation(mol_dataset, mol_evaluator):
    LLM = PromptedLLM.from_name("gpt2", backend="hf")

    def sampler_factory(instance):
        LLM.prompt_ids = default_prompt_formatter(LLM.model.tokenizer, instance)
        return direct_token_sampler(LLM)

    def critic_factory(instance):
        return PartialSMILES().coerce(LLM, f=b"".join)

    async def model(instance, output_dir, replicate):
        assert replicate == 0

        sequences = await sampler_factory(instance).smc(
            critic=critic_factory(instance),
            n_particles=2,
            ess_threshold=0.5,
            max_tokens=10,
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
            dataset=mol_dataset,
            evaluator=mol_evaluator,
            model=model,
            n_replicates=1,
        )
    )

    assert result["n_instances"] == 2
