import re
from typing import List, Optional
import planetarium

from genlm.eval.core import Evaluator, EvaluationResult, Instance, Dataset
from datasets import load_dataset

##########################
#               Dataset  #
##########################


class GoalInferenceInstance(Instance):
    """Schema for a single Planetarium goal-inference item."""

    nl_goal: str
    problem_text: str
    masked_pddl: str
    prefix_pddl: str
    domain_name: str

    def __str__(self):
        return (
            f"GoalInferenceInstance(id={self.instance_id}, domain={self.domain_name})"
        )


class GoalInferenceDataset(Dataset[GoalInferenceInstance]):
    """Dataset wrapper yielding GoalInferenceInstance items."""

    def __init__(self, dev_items: List[dict]):
        """Store preprocessed records."""
        self.dev_items = dev_items

    @staticmethod
    def _make_prefix_pddl(problem_text: str) -> Optional[str]:
        """Return text up to and including '(:goal (and' for prompting.

        Args:
            problem_text: Full PDDL problem text.

        Returns:
            Prefix string or None if the pattern is absent.
        """
        m = re.search(r"\(:goal\s*\(and", problem_text)
        if not m:
            return None
        return problem_text[: m.end()]

    @staticmethod
    def _mask_goal_for_reference(problem_text: str) -> Optional[str]:
        """Create a masked PDDL with '[BLANK]' in place of the goal.

        Args:
            problem_text: Full PDDL problem text.

        Returns:
            Masked PDDL or None if no goal section is found.
        """
        i = problem_text.find("(:goal")
        if i == -1:
            return None
        prefix_before_goal = problem_text[:i]
        goal_suffix = "(:goal (and [BLANK]))\n)"
        return prefix_before_goal + goal_suffix

    def __iter__(self):
        """Yield GoalInferenceInstance objects built from stored records."""
        for i, rec in enumerate(self.dev_items):
            problem_text = rec["problem_text"]
            prefix_pddl = self._make_prefix_pddl(problem_text)
            masked_pddl = self._mask_goal_for_reference(problem_text)
            if prefix_pddl is None or masked_pddl is None:
                continue

            yield GoalInferenceInstance(
                nl_goal=rec["nl_goal"],
                problem_text=problem_text,
                masked_pddl=masked_pddl,
                prefix_pddl=prefix_pddl,
                instance_id=rec.get("instance_id", i),
                domain_name=rec["domain_name"],
            )

    @classmethod
    def from_hf_planetarium(
        cls,
        split: str = "train",
        subset: str = "default",
        max_objects: int = 10,
        domains: Optional[List[str]] = None,
    ) -> "GoalInferenceDataset":
        """Load and filter Planetarium data via HuggingFace Datasets.

        Args:
            split: Dataset split to load (e.g., "train", "validation", "test").
            subset: Named configuration (Planetarium subset).
            max_objects: Keep problems with at most this many objects.
            domains: Optional list of domain names to include (case-insensitive).

        Returns:
            GoalInferenceDataset with filtered instances.
        """
        ds = load_dataset("BatsResearch/planetarium", name=subset, split=split)

        allowed = None if domains is None else {d.lower() for d in domains}
        dev_items: List[dict] = []

        for ex in ds:
            dom = str(ex["domain"]).lower()
            if allowed is not None and dom not in allowed:
                continue
            if int(ex.get("num_objects", 0)) > int(max_objects):
                continue

            dev_items.append(
                {
                    "instance_id": int(ex.get("id", len(dev_items))),
                    "nl_goal": str(ex["natural_language"]),
                    "problem_text": str(ex["problem_pddl"]),
                    "domain_name": dom,
                }
            )

        return cls(dev_items)

    @property
    def schema(self):
        return GoalInferenceInstance


##########################
#       Evaluator        #
##########################


class GoalInferenceEvaluator(Evaluator[GoalInferenceInstance]):
    """Evaluator using Planetarium equivalence on masked PDDL reconstruction."""

    def evaluate_sample(
        self, instance: GoalInferenceInstance, response: str
    ) -> EvaluationResult:
        """Inject prediction into masked PDDL and check equivalence.

        Args:
            instance (GoalInferenceInstance): The goal-inference item being evaluated.
            response (str): Model output to splice into the goal (no closing paren).

        Returns:
            EvaluationResult with score 1.0 if equivalent, else 0.0.
        """
        masked = instance.masked_pddl
        full_pddl = instance.problem_text
        if not masked or not full_pddl:
            return EvaluationResult(score=0.0, desc="missing_problem_or_masked")

        if "[BLANK]" not in masked:
            return EvaluationResult(score=0.0, desc="no_blank_marker")

        pred = response.strip() if response is not None else ""
        generated_pddl = masked.replace("[BLANK]", pred + ")")  # Add missing bracket
        try:
            ok = planetarium.evaluate(full_pddl, generated_pddl)[2]
        except (ValueError, AttributeError):
            return EvaluationResult(score=0.0, desc="planetarium_error")

        return EvaluationResult(
            score=1.0 if ok else 0.0,
            desc="equiv" if ok else "not_equiv",
            metadata={"candidate": generated_pddl},
        )


###############
# Model Utils #
###############

GOAL_SYSTEM_PROMPT = (
    "You are a PDDL expert, who writes valid PDDL code that "
    "describes user-provided planning problems directly without further "
    "explanations or texts.\n\n"
)


def goal_default_prompt_formatter(
    tokenizer,
    instance: GoalInferenceInstance,
    use_chat_format: bool = False,
    system_prompt: str = GOAL_SYSTEM_PROMPT,
):
    """Format the prompt to reproduce the reference assistant-prefix prompting.

    Args:
        tokenizer (Tokenizer): The tokenizer to use.
        instance (GoalInferenceInstance): The instance to format.
        use_chat_format (bool): Whether to use chat format.
        system_prompt (str): The system prompt to use.

    Returns:
        (list[int]): The prompt ids.
    """
    if use_chat_format:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Natural Language goal description: \n\n"
                + instance.nl_goal
                + "\n\n",
            },
            {"role": "assistant", "content": instance.prefix_pddl},
        ]
        return tokenizer.apply_chat_template(
            conversation=messages, tokenize=True, add_generation_prompt=True
        )

    prompt = (
        system_prompt
        + "Natural Language goal description: \n\n"
        + instance.nl_goal
        + "\n\n"
        + instance.prefix_pddl
    )
    return tokenizer.encode(prompt)
