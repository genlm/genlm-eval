import regex
import string
from genlm.control import PromptedLLM, Potential
from genlm.eval.models.control import ControlModelAdaptor


class PatternPotential(Potential):
    """Potential function for regex pattern matching."""

    def __init__(self, pattern):
        vocab = list(map(ord, string.printable))
        super().__init__(vocab)
        self.r = regex.compile(pattern)

    async def complete(self, context):
        text = "".join(map(chr, context))
        match = self.r.fullmatch(text) is not None
        return 0.0 if match else float("-inf")

    async def prefix(self, context):
        text = "".join(map(chr, context))
        m = self.r.match(text, partial=True)
        match = m is not None and m.start() == 0 and m.end() == len(text)
        return 0.0 if match else float("-inf")


class PatternMatchingModel(ControlModelAdaptor):
    """Model adaptor for pattern matching."""

    def make_llm(self, model_name, lm_args):
        """Make the LLM for the model."""
        llm = PromptedLLM.from_name(model_name, **lm_args)
        eos = [t for t in llm.vocab if b"\n" in t] + [
            llm.model.tokenizer.eos_token.encode("utf-8")
        ]
        return llm.spawn_new_eos(eos)

    def make_prompt_ids(self, instance):
        """Make the prompt ids for the model."""
        few_shot_examples = [
            ("(ab)+", "ab"),
            ("(ab|cd)+", "cd"),
            ("[a-z]+", "hello"),
        ]

        system_prompt = (
            "You are a helpful assistant that generates strings matching regular expressions. "
            + "Only output the exact string that matches the regex pattern, nothing more."
        )

        return self.llm.model.tokenizer.apply_chat_template(
            self.chat_template_formatter(
                system_prompt, few_shot_examples, instance.pattern
            ),
            tokenize=True,
            add_generation_prompt=True,
        )
