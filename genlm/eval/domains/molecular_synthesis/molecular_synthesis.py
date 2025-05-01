import random
import numpy as np
from functools import lru_cache
from genlm.control import Potential

from genlm.eval.core import EvaluationResult, Instance, Dataset, Evaluator


###########
# Dataset #
###########


class MolecularSynthesisInstance(Instance):
    """Schema for molecular synthesis instance."""

    molecules: list[str]


class MolecularSynthesisDataset(Dataset[MolecularSynthesisInstance]):
    """Dataset for molecular synthesis evaluation."""

    def __init__(self, prompt_molecules):
        """Initialize the dataset with a list of molecules.

        Args:
            prompt_molecules: List of lists of molecules which will be used to generate prompts.
        """
        self.prompt_molecules = prompt_molecules

    def __len__(self):
        return len(self.prompt_molecules)

    @classmethod
    def from_smiles(cls, smiles_path, n_molecules=20, n_instances=100, seed=1234):
        """Load molecules from a SMILES file.

        Args:
            smiles_path (str): Path to the .smi file containing SMILES strings.
            n_molecules (int): Number of molecules to sample.
            n_instances (int): Number of instances to sample.
            seed (int): Seed for the random number generator.

        Returns:
            MolecularSynthesisDataset: Dataset initialized with molecules from the SMILES.
        """
        molecules = open(smiles_path).readlines()
        prompt_molecules = []
        random.seed(seed)
        for _ in range(n_instances):
            molecule_ids = random.sample(range(len(molecules)), n_molecules)
            prompt_molecules.append([molecules[i] for i in molecule_ids])
        return cls(prompt_molecules)

    def __iter__(self):
        """Iterate over molecules.

        Returns:
            Iterator[MolecularSynthesisInstance]: Iterator over molecular synthesis instances.
        """
        for i, molecules in enumerate(self.prompt_molecules):
            yield MolecularSynthesisInstance(molecules=molecules, instance_id=i)

    @property
    def schema(self):
        """Get the schema class for this dataset.

        Returns:
            type[MolecularSynthesisInstance]: The Pydantic model class for molecular synthesis instances.
        """
        return MolecularSynthesisInstance


#############
# Evaluator #
#############


def get_mol(smiles):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        Chem.Kekulize(mol)
    return mol


@lru_cache
def cached_eval(mol):
    from rdkit.Chem import QED

    mol = get_mol(mol)
    if mol is None:
        return False, 0.0
    return True, QED.qed(mol=mol)


class MolecularSynthesisEvaluator(Evaluator[MolecularSynthesisInstance]):
    """Evaluator for molecular synthesis."""

    def evaluate_sample(self, instance, response):
        """Evaluate if a response matches the regex pattern.

        Args:
            instance (PatternMatchingInstance): The pattern matching instance being evaluated.
            response (str): The model's response text.

        Returns:
            (bool): Whether the response matches the pattern.
        """
        valid, acc = cached_eval(response.strip())
        desc = "valid" if valid else "invalid"
        return EvaluationResult(score=acc, desc=desc)


###############
# Model Utils #
###############


SYSTEM_PROMPT = (
    "You are an expert in chemistry. "
    "You are given a list of molecules in SMILES format. "
    "You are asked to write another molecule in SMILES format with similar chemical properties."
)


class PartialSMILES(Potential):
    def __init__(self):
        super().__init__(vocabulary=list(range(256)))

    async def prefix(self, context):
        string = bytes(context).decode("utf-8", errors="ignore")
        if len(string) > 0 and string[0] == " ":
            string = string[1:]
        return self._validate(string, partial=True)

    async def complete(self, context):
        string = bytes(context).decode("utf-8", errors="ignore")
        if len(string) > 0 and string[0] == " ":
            string = string[1:]
        return self._validate(string, partial=False)

    def _validate(self, smiles, partial):
        import partialsmiles as ps

        try:
            ps.ParseSmiles(smiles, partial=partial)
            return 0.0
        except Exception:
            return -np.inf
