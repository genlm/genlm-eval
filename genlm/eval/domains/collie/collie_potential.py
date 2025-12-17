import logging
from typing import Any

import numpy as np
from genlm.control import Potential


log = logging.getLogger(__name__)


class CollieConstraintPotential(Potential):
    """Potential that guides generation towards satisfying Collie constraints.

    This potential evaluates partial and complete text against a Collie constraint
    to guide the language model during generation. It uses the constraint's extract()
    method to check progress and penalizes clear violations.

    Strategy:
    - For count constraints (words, characters, sentences, etc.):
      - During generation, allow text if count is below target
      - Penalize heavily if count exceeds target by too much
    - For required content (keywords, phrases):
      - Check if required content is present
      - Be lenient during partial generation
    - For structural constraints (paragraphs, sentences):
      - Check structure incrementally

    Args:
        constraint: The Collie constraint object from collie-bench
        targets: The target values for the constraint
        tolerance: Multiplicative tolerance for count-based constraints (default: 1.5)
                   E.g., if target is 100 words, allow up to 150 words during generation TODO

    Example:
        >>> from collie import constraints as cc
        >>> constraint = cc.Constraint(
        ...     cc.InputLevel(None),
        ...     cc.TargetLevel("word"),
        ...     cc.Count(),
        ...     cc.Relation("=="),
        ...     cc.Reduction(None),
        ... )
        >>> potential = CollieConstraintPotential(constraint, targets=50)
    """

    def __init__(
        self,
        constraint: Any,
        targets: Any,
        tolerance: float = 1.5,
        verbose: bool = False,
    ):
        super().__init__(vocabulary=list(range(256)))
        self.constraint = constraint
        self.targets = targets
        self.tolerance = tolerance
        self.verbose = verbose
        self._constraint_info = self._analyze_constraint()

    def _analyze_constraint(self):
        """Analyze the constraint to determine how to evaluate partial text."""
        info = {
            "type": "unknown",
            "is_count": False,
            "is_foreach": False,
            "is_relation": False,
        }

        try:
            # Check if it's a count-based constraint
            if hasattr(self.constraint, "transformation"):
                trans_str = str(self.constraint.transformation)
                if "Count()" in trans_str:
                    info["is_count"] = True
                    info["type"] = "count"
                elif "ForEach" in trans_str:
                    info["is_foreach"] = True
                    info["type"] = "foreach"

            # Check relation type
            if hasattr(self.constraint, "relation"):
                rel_str = str(self.constraint.relation)
                if "==" in rel_str:
                    info["is_relation"] = "equal"
                elif "<=" in rel_str:
                    info["is_relation"] = "less_equal"
                elif ">=" in rel_str:
                    info["is_relation"] = "greater_equal"
                elif "in" in rel_str.lower():
                    info["is_relation"] = "contains"
        except Exception as e:
            if self.verbose:
                log.warning(f"Could not analyze constraint: {e}")

        return info

    async def prefix(self, context):
        """Evaluate partial text during generation.

        Returns:
            0.0 if the partial text is acceptable (on track or unclear)
            -inf if the partial text clearly violates the constraint
        """
        string = bytes(context).decode("utf-8", errors="ignore")
        return self._evaluate_partial(string)

    async def complete(self, context):
        """Evaluate complete text at the end of generation.

        Returns:
            0.0 if the text satisfies the constraint
            -inf if the text violates the constraint
        """
        string = bytes(context).decode("utf-8", errors="ignore")
        return self._evaluate_complete(string)

    def _evaluate_partial(self, text: str) -> float:
        """Evaluate partial text during generation."""
        if not text or not text.strip():
            return 0.0

        try:
            extracted = self.constraint.extract(text)
            # For count-based constraints
            if self._constraint_info["is_count"]:
                current_count = (
                    extracted
                    if isinstance(extracted, (int, float))
                    else len(extracted) if isinstance(extracted, list) else 0
                )
                target_value = (
                    self.targets
                    if isinstance(self.targets, (int, float))
                    else (
                        self.targets[0]
                        if isinstance(self.targets, list)
                        else float("inf")
                    )
                )
                # Allow text if we're below or near target
                if self._constraint_info["is_relation"] == "equal":
                    # For ==, allow if we haven't exceeded target by too much
                    if current_count <= target_value * self.tolerance:
                        return 0.0
                    else:
                        return -np.inf  # Clearly exceeded target

                elif self._constraint_info["is_relation"] == "less_equal":
                    # For <=, penalize if we've exceeded target
                    if current_count <= target_value * self.tolerance:
                        return 0.0
                    else:
                        return -np.inf

                elif self._constraint_info["is_relation"] == "greater_equal":
                    # For >=, always allow during generation
                    return 0.0
            # For ForEach constraints (e.g., each word <= N characters)
            elif self._constraint_info["is_foreach"]:
                if isinstance(extracted, list) and isinstance(self.targets, list):
                    # Only penalize clear violations
                    for i, val in enumerate(extracted[-3:]):
                        target_idx = min(i, len(self.targets) - 1)
                        target_val = self.targets[target_idx]
                        if self._constraint_info["is_relation"] == "equal":
                            if abs(val - target_val) > target_val * 0.5:
                                return -np.inf
                        elif self._constraint_info["is_relation"] == "less_equal":
                            if val > target_val * self.tolerance:
                                return -np.inf
                        elif self._constraint_info["is_relation"] == "greater_equal":
                            pass
                return 0.0
            return 0.0

        except Exception as e:
            if self.verbose:
                log.debug(f"Error evaluating partial text: {e}")
            return 0.0

    def _evaluate_complete(self, text: str) -> float:
        """Evaluate complete text using the constraint's check() method."""
        try:
            # Use the official Collie constraint checking
            is_valid = self.constraint.check(text, self.targets)
            return 0.0 if is_valid else -np.inf
        except Exception as e:
            if self.verbose:
                log.error(f"Error evaluating complete text: {e}")
            return -np.inf
