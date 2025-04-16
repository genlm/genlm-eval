from genlm.eval.core import Evaluator, EvaluationResult
from .dataset import MolecularSynthesisInstance

from functools import lru_cache


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

    def evaluate_response(self, instance, response):
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
