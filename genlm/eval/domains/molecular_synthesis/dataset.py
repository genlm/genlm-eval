import random

from genlm.eval.core import Dataset, Instance


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
