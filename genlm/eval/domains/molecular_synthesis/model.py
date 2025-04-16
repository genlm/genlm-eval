import os
import numpy as np
from functools import lru_cache, cached_property

from genlm.control import Potential, PromptedLLM, BoolCFG
from genlm.eval.models.control import ControlModelAdaptor


class PartialSMILES(Potential):
    def __init__(self):
        super().__init__(vocabulary=list(range(256)))

    @lru_cache(maxsize=None)
    def _parse(self, query):
        return self.parser.parse(query)

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
        import ps

        try:
            ps.ParseSmiles(smiles, partial=partial)
            return 0.0
        except Exception:
            return -np.inf


class MolecularSynthesisModel(ControlModelAdaptor):
    """Model adaptor for molecular synthesis."""

    instruction = "You are an expert in chemistry. You are given a list of molecules in SMILES format. You are asked to write another molecule in SMILES format with similar chemical properties.\n"
    prediction_prompt = "Molecule:"

    def exemplar_formatter(self, molecule):
        return f"Molecule: {molecule}"

    def __init__(self, *, grammar_path=None, **kwargs):
        super().__init__(**kwargs)

        if grammar_path is None:
            grammar_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "data/smiles.lark"
            )

        if not os.path.exists(grammar_path):
            raise FileNotFoundError(f"Grammar file not found: {grammar_path}")

        with open(grammar_path, "r") as f:
            self.grammar = f.read()

    @cached_property
    def bool_cfg(self):
        return BoolCFG.from_lark(self.grammar)

    def make_llm(self, model_name, lm_args):
        """Make the LLM for the model."""
        llm = PromptedLLM.from_name(model_name, **lm_args)
        eos_tokens = [t for t in llm.vocab if b"\n" in t]
        return llm.spawn_new_eos(eos_tokens)

    def make_prompt_ids(self, instance):
        """Make the prompt ids for the model."""
        prompt = (
            self.instruction
            + "".join([self.exemplar_formatter(m) for m in instance.molecules])
            + self.prediction_prompt
        )
        return self.llm.model.tokenizer.encode(prompt)

    def metadata(self):
        base_metadata = super().metadata()
        base_metadata["grammar"] = self.grammar
        return base_metadata
