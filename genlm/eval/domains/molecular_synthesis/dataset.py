import random

from genlm.eval.core import Dataset, Instance


class MolecularSynthesisInstance(Instance):
    """Schema for molecular synthesis instance."""

    molecules: list[str]


class MolecularSynthesisDataset(Dataset[MolecularSynthesisInstance]):
    """Dataset for molecular synthesis evaluation."""

    def __init__(self, molecules, n_molecules=20, n_prompts=100, seed=1234):
        """Initialize the dataset with a list of molecules.

        Args:
            molecules: List of molecules to evaluate.
            n_molecules: Number of molecules to sample.
            n_prompts: Number of prompts to sample.
            seed: Seed for the random number generator.
        """
        self.molecules = molecules
        self.n_molecules = n_molecules
        self.n_prompts = n_prompts
        self.seed = seed
        random.seed(seed)

    @classmethod
    def from_smiles(cls, smiles_path, n_molecules=20, n_prompts=100, seed=1234):
        """Load molecules from a SMILES file.

        Args:
            smiles_path (str): Path to the .smi file containing SMILES strings.
            n_molecules (int): Number of molecules to sample.
            n_prompts (int): Number of prompts to sample.
            seed (int): Seed for the random number generator.

        Returns:
            MolecularSynthesisDataset: Dataset initialized with molecules from the SMILES.
        """
        molecules = open(smiles_path).readlines()
        return cls(molecules, n_molecules, n_prompts, seed)

    def __iter__(self):
        """Iterate over molecules.

        Returns:
            Iterator[MolecularSynthesisInstance]: Iterator over molecular synthesis instances.
        """
        for instance_id in range(self.n_prompts):
            molecule_ids = random.sample(range(len(self.molecules)), self.n_molecules)
            molecules = [self.molecules[i] for i in molecule_ids]
            yield MolecularSynthesisInstance(
                molecules=molecules, instance_id=instance_id
            )

    @property
    def schema(self):
        """Get the schema class for this dataset.

        Returns:
            type[MolecularSynthesisInstance]: The Pydantic model class for molecular synthesis instances.
        """
        return MolecularSynthesisInstance
