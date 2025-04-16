from genlm.eval.core import Evaluator, EvaluationResult
from .dataset import MolecularSynthesisInstance

from functools import lru_cache


class HiddenPrints:
    """Context manager to disable RDKit logs. By default all logs are disabled."""

    def __init__(
        self,
        mute_errors: bool = True,
        mute_warning: bool = True,
        mute_info: bool = True,
        mute_debug: bool = True,
    ):
        # Get current log state
        self.previous_status = self._get_log_status()

        # Init the desired log state to apply during in the context
        self.desired_status = {}
        self.desired_status["rdApp.error"] = not mute_errors
        self.desired_status["rdApp.warning"] = not mute_warning
        self.desired_status["rdApp.debug"] = not mute_debug
        self.desired_status["rdApp.info"] = not mute_info

    def _get_log_status(self):
        """Get the current log status of RDKit logs."""
        from rdkit import rdBase

        log_status = rdBase.LogStatus()
        log_status = {
            st.split(":")[0]: st.split(":")[1] for st in log_status.split("\n")
        }
        log_status = {
            k: True if v == "enabled" else False for k, v in log_status.items()
        }
        return log_status

    def _apply_log_status(self, log_status):
        """Apply an RDKit log status."""
        from rdkit import rdBase

        for k, v in log_status.items():
            if v is True:
                rdBase.EnableLog(k)
            else:
                rdBase.DisableLog(k)

    def __enter__(self):
        self._apply_log_status(self.desired_status)

    def __exit__(self, *args, **kwargs):
        self._apply_log_status(self.previous_status)


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
