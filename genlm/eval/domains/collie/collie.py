import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Type
import urllib.request

from genlm.eval.core import EvaluationResult, Instance, Dataset, Evaluator

log = logging.getLogger(__name__)

################
# Collie Data  #
################


class CollieInstance(Instance):
    """Schema for a Collie instance."""

    prompt: str
    example: str
    targets: Any
    metadata: Dict[str, Any]
    constraint_type: str
    constraint: Any  # The actual Collie constraint object


class CollieDataset(Dataset[CollieInstance]):
    """Dataset for Collie evaluation (Yao et al., 2023)."""

    def __init__(self, rows: List[Mapping[str, Any]]):
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[CollieInstance]:
        for i, row in enumerate(self._rows):
            metadata = row.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            yield CollieInstance(
                prompt=str(row.get("prompt", "")).strip(),
                example=str(row.get("example", "")).strip(),
                targets=row.get("targets"),
                metadata=metadata,
                constraint_type=row.get("constraint_type", ""),
                constraint=row.get("constraint"),
                instance_id=i,
            )

    @property
    def schema(self) -> Type[CollieInstance]:
        return CollieInstance

    @classmethod
    def from_official(
        cls,
        constraint_types: Optional[Sequence[str]] = None,
        max_instances: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 1234,
        cache_dir: Optional[str] = None,
    ) -> "CollieDataset":
        """Load Collie from the official Princeton-NLP repository.

        Downloads and loads the official all_data.dill file from:
        https://github.com/princeton-nlp/Collie

        Args:
            constraint_types: Optional list of constraint types to filter by
            max_instances: Maximum number of instances to load
            shuffle: Whether to shuffle the dataset
            seed: Random seed for shuffling
            cache_dir: Optional cache directory for the dill file

        Returns:
            CollieDataset instance
        """
        try:
            import dill
        except ImportError:
            raise ImportError(
                "dill is required to load the official Collie dataset. "
                "Install with: pip install dill"
            )

        # Set up cache directory
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "genlm-eval" / "collie"
        else:
            cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Download if not cached
        dill_path = cache_dir / "all_data.dill"
        if not dill_path.exists():
            log.info("Downloading official Collie dataset from GitHub...")
            url = (
                "https://github.com/princeton-nlp/Collie/raw/master/data/all_data.dill"
            )
            urllib.request.urlretrieve(url, dill_path)
            log.info(f"Downloaded to {dill_path}")

        # Load the dill file
        log.info(f"Loading Collie dataset from {dill_path}")
        with open(dill_path, "rb") as f:
            data = dill.load(f)

        # Flatten the data structure (dict of lists -> single list)
        rows = []
        for constraint_type, instances in data.items():
            for instance in instances:
                instance["constraint_type"] = constraint_type
                rows.append(instance)

        # Filter by constraint types
        constraint_set = (
            {x.lower() for x in constraint_types} if constraint_types else None
        )
        if constraint_set:
            rows = [
                r
                for r in rows
                if r.get("constraint_type", "").lower() in constraint_set
            ]

        # Shuffle if requested
        if shuffle:
            rnd = random.Random(seed)
            rnd.shuffle(rows)

        # Limit instances
        if isinstance(max_instances, int) and max_instances >= 0:
            rows = rows[:max_instances]

        log.info("Loaded Collie: %d instances from official dataset", len(rows))
        return cls(rows)


################
#   Evaluator  #
################


class CollieEvaluator(Evaluator[CollieInstance]):
    """Evaluator for Collie constrained text generation tasks.

    Uses the Collie library's Constraint.check() API to evaluate constraints.
    """

    def __init__(self):
        """Initialize the Collie evaluator."""
        # Verify collie-bench is installed
        try:
            import collie  # type: ignore # noqa: F401
        except ImportError:
            raise ImportError(
                "collie-bench library is required. Install with: pip install collie-bench"
            )

    def evaluate_sample(
        self, instance: CollieInstance, response: str
    ) -> EvaluationResult:
        """Evaluate if a response satisfies the constraints using Collie's Constraint.check() API.

        Args:
            instance: The Collie instance being evaluated
            response: The model's generated text response

        Returns:
            EvaluationResult with score (1.0 if constraints satisfied, 0.0 otherwise)
        """
        if not response or not response.strip():
            return EvaluationResult(
                score=0.0, desc="empty response", metadata=instance.metadata
            )

        response = response.strip()

        # Use the constraint object directly - no string parsing needed!
        try:
            is_valid = bool(instance.constraint.check(response, instance.targets))
        except Exception as e:
            log.error(
                f"Error during constraint check for instance {instance.instance_id}: {e}"
            )
            is_valid = False

        desc = f"constraint_type={instance.constraint_type}, valid={is_valid}"
        return EvaluationResult(
            score=1.0 if is_valid else 0.0,
            desc=desc,
            metadata={**instance.metadata, "constraint_type": instance.constraint_type},
        )


##########################
# Prompt formatter (LM)  #
##########################


def default_prompt_formatter(
    tokenizer,
    instance: CollieInstance,
    use_chat_format: bool = False,
) -> List[int]:
    """Default prompt formatter for Collie.

    Args:
        tokenizer: The tokenizer to use
        instance: The Collie instance to format
        use_chat_format: Whether to use chat format

    Returns:
        List of token IDs
    """
    if use_chat_format:
        chat_template = getattr(tokenizer, "chat_template", None)
        if chat_template:
            messages = [{"role": "user", "content": instance.prompt}]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        # Fallback: treat as plain prompt formatting.
        return tokenizer.encode(instance.prompt)
    else:
        return tokenizer.encode(instance.prompt)
