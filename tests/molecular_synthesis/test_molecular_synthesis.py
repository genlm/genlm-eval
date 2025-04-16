import pytest
import asyncio
from pathlib import Path

from genlm.control import direct_token_sampler

from genlm.eval.domains.molecular_synthesis.model import MolecularSynthesisModel
from genlm.eval.domains.molecular_synthesis.dataset import (
    MolecularSynthesisDataset,
    MolecularSynthesisInstance,
)
from genlm.eval.domains.molecular_synthesis.evaluator import MolecularSynthesisEvaluator

from genlm.eval.core import ModelOutput, ModelResponse, run_evaluation


@pytest.fixture
def mol_dataset():
    smiles_path = Path(__file__).parent / "data/test_molecules.smi"
    return MolecularSynthesisDataset.from_smiles(
        smiles_path, n_molecules=2, n_prompts=2
    )


@pytest.fixture
def mol_evaluator():
    return MolecularSynthesisEvaluator()


def test_mol_data(mol_dataset):
    assert mol_dataset.schema is MolecularSynthesisInstance
    for instance in mol_dataset:
        assert isinstance(instance, MolecularSynthesisInstance)


def test_mol_evaluator(mol_dataset, mol_evaluator):
    first_instance = next(iter(mol_dataset))
    valid_smiles = "BrC1=C2C3C4C3N(CC4C#C)C2=NC(=O)S1"

    # Test with a valid SMILES string (benzene)
    result = mol_evaluator.evaluate_response(first_instance, valid_smiles)
    assert result.score > 0
    assert result.desc == "valid"

    ensemble_result = mol_evaluator.evaluate_ensemble(
        first_instance,
        ModelOutput(
            responses=[
                ModelResponse(text=valid_smiles, prob=0.5),
                ModelResponse(text=valid_smiles, prob=0.5),
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
                ModelResponse(text="invalid_smiles", prob=0.5),
                ModelResponse(text=valid_smiles, prob=0.5),
            ],
            runtime_seconds=0.1,
        ),
    )

    assert 0.0 < ensemble_result["weighted_accuracy"] <= 0.5
    assert ensemble_result["runtime_seconds"] == 0.1


def test_mol_model_init():
    model = MolecularSynthesisModel(
        model_name="placeholder",  # should be ignored so long as we don't access .llm
        max_tokens=100,
        n_particles=2,
        ess_threshold=0.5,
    )

    assert model.max_tokens == 100
    assert model.n_particles == 2
    assert model.ess_threshold == 0.5
    assert model.bool_cfg is not None


def test_run_evaluation(mol_dataset, mol_evaluator):
    class TestMolModel(MolecularSynthesisModel):
        def make_sampler(self, instance):
            return direct_token_sampler(self.llm)

        def make_critic(self, instance):
            return

    n_particles = 2

    model = TestMolModel(
        model_name="gpt2",
        max_tokens=100,
        n_particles=n_particles,
        ess_threshold=0.5,
    )

    n_replicates = 1

    result = asyncio.run(
        run_evaluation(
            dataset=mol_dataset,
            evaluator=mol_evaluator,
            model=model,
            n_replicates=n_replicates,
        )
    )

    assert result["n_instances"] == 2
