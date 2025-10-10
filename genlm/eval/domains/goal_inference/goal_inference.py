import re
from typing import List, Optional
import planetarium
import polars as pl
from huggingface_hub import hf_hub_download

from genlm.eval.core import Evaluator, EvaluationResult, Instance, Dataset
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
        n_examples: int = 100,
        max_objects: int = 9,
        shard_filename: str = "data/train-00000-of-00001.parquet",
        domains: Optional[List[str]] = None,
    ) -> "GoalInferenceDataset":
        """Load and filter Planetarium data via HuggingFace Datasets.

        Args:
            n_examples: Number of instances to evaluate.
            max_objects: Keep problems with at most this many objects.
            shard_filename: Specific shard file to download from Planetarium.
            domains: Optional list of domain names to include (case-insensitive).

        Returns:
            GoalInferenceDataset with filtered instances.
        """
        local_path = hf_hub_download(
            repo_id="BatsResearch/planetarium",
            repo_type="dataset",
            filename=shard_filename,
        )

        allowed = {"blocksworld"} if domains is None else {d.lower() for d in domains}

        df = pl.read_parquet(local_path)
        df = (
            df.with_columns(
                problem_pddl=pl.col("problem_pddl").str.replace("\r\n", "\n", literal=True),
                natural_language=pl.col("natural_language").fill_null(""),
            )
            .filter(pl.col("problem_pddl").str.contains("(:goal (and", literal=True))
            .with_columns(
                goal_natural_language=pl.concat_str(
                    pl.lit("Your goal"),
                    pl.col("natural_language").str.split(by="Your goal").list.last(),
                ),
            )
            .filter(
                (pl.col("domain").str.to_lowercase().is_in(list(allowed)))
                & (pl.col("num_objects") <= max_objects)
                & (pl.col("init_is_abstract") == 0)
                & (pl.col("goal_is_abstract") == 0)
            )
            .unique(subset=["goal_natural_language"])
            .head(n_examples)
            .select(
                pl.col("id").alias("instance_id"),
                pl.col("goal_natural_language").alias("nl_goal"),
                pl.col("problem_pddl").alias("problem_text"),
                pl.col("domain").str.to_lowercase().alias("domain_name"),
            )
        )

        items = df.to_dicts()
        return cls(items)

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
