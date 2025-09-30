import re
import shlex
import tempfile
import hashlib
import subprocess
from pathlib import Path

from genlm.control import Potential

class GoalInferenceVALPotential(Potential):
    """Expensive potential that validates partial goal strings with VAL.

    It splices the current candidate goal into the problem, reuses a (cached)
    Fast-Downward plan for (domain, problem), and returns 0.0 if VAL
    validates the plan under the candidate goal (-inf otherwise).
    """
    def __init__(
        self,
        domain_pddl_text: str,
        problem_pddl_text: str,
        fast_downward_cmd: str = "./fast-downward.sif",
        val_cmd: str = "Validate",
        cache_root: Path | str = "benchmark/goal_inference/data",
        verbosity: int = 0,
    ):
        super().__init__(list(range(256)))
        self.domain = domain_pddl_text.replace("\r\n", "\n").replace("\r", "\n")
        self.problem = problem_pddl_text.replace("\r\n", "\n").replace("\r", "\n")
        self.fast_downward_cmd = fast_downward_cmd
        self.val_cmd = val_cmd
        self.verbosity = verbosity

        self.cache_root = Path(cache_root)
        self.tasks_dir = self.cache_root / "pddl_tasks"
        self.plans_dir = self.cache_root / "pddl_plans"
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.plans_dir.mkdir(parents=True, exist_ok=True)

        # Hash both domain + problem to key plan cache
        key = hashlib.sha256((self.domain + "\n---\n" + self.problem).encode("utf-8")).hexdigest()
        self.task_path = self.tasks_dir / f"{key}.pddl"
        self.plan_path = self.plans_dir / f"{key}.pddl"
        self.domain_path = self.tasks_dir / f"{key}.domain.pddl"

        if not self.task_path.exists():
            self.task_path.write_text(self.problem, encoding="utf-8")
        if not self.domain_path.exists():
            self.domain_path.write_text(self.domain, encoding="utf-8")
        if not self.plan_path.exists():
            self._ensure_plan()

    def _ensure_plan(self):
        """Generate and cache a plan with Fast-Downward if missing.

        Raises:
            ValueError: If Fast-Downward exits with a non-zero status.
        """
        cmd = (
            f"{shlex.quote(self.fast_downward_cmd)} "
            f"--plan-file {shlex.quote(str(self.plan_path))} "
            f"{shlex.quote(str(self.domain_path))} {shlex.quote(str(self.task_path))} "
            f'--search "astar(ipdb())"'
        )
        proc = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, encoding="utf-8")
        if proc.returncode != 0:
            raise ValueError(f"[Fast-Downward] nonzero exit:\n{proc.stderr}")

    @staticmethod
    def _fix_context(s: str) -> str | None:
        """Heuristically close extra parentheses to get a parsable goal prefix.

        Args:
            s: Current decoded context string.

        Returns:
            A version with the last balanced ')' ensured, or None if none found.
        """
        s = s.replace(")))\n", ")").replace(")))", ")").replace("))", ")")
        head, sep, tail = s.rpartition(")")
        if not head:
            return None
        return head + ")"

    @staticmethod
    def _splice_goal(problem_pddl: str, ctx: str) -> str:
        """Replace the goal in `problem_pddl` with `(and {ctx})`.

        Args:
            problem_pddl (str): Full problem text.
            ctx (str): Goal inner content to inject after `(and `.

        Returns:
            str: Modified problem PDDL with the new goal.
        """
        return re.sub(r"\(:goal \(and .*?\)\)\n", f"(:goal (and {ctx}))\n", problem_pddl, flags=re.DOTALL)

    def _energy(self, s: str) -> float:
        """Compute potential: 0.0 if VAL validates, else -inf.

        Args:
            s: Current (partial) goal text emitted by the model.

        Returns:
            0.0 on validation success, -inf otherwise (or if unparsable).
        """
        parsable = self._fix_context(s)
        if parsable is None:
            return 0.0
        gen = self._splice_goal(self.problem, parsable)
        with tempfile.TemporaryDirectory(prefix="val_goal_") as tmp:
            tmp_problem = Path(tmp) / "task.pddl"
            tmp_problem.write_text(gen, encoding="utf-8")
            cmd = f"{shlex.quote(self.val_cmd)} {shlex.quote(str(self.domain_path))} {shlex.quote(str(tmp_problem))} {shlex.quote(str(self.plan_path))}"
            proc = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, encoding="utf-8")
            return 0.0 if proc.returncode == 0 else float("-inf")

    async def prefix(self, context: bytes) -> float:
        """Score a partial prefix during generation.

        Args:
            context (bytes): Byte prefix generated so far.

        Returns:
            float: Potential value for the prefix (0.0 / -inf).
        """
        try: 
            return self._energy(context.decode("utf-8"))
        except Exception: 
            return float("-inf")

    async def complete(self, context: bytes) -> float:
        """Score a completed sequence at EOS.

        Args:
            context: Final byte sequence (without our trailing ')' heuristic).

        Returns:
            Potential value for the complete string (0.0 / -inf).
        """
        try: 
            return self._energy(context.decode("utf-8") + ")")
        except Exception: 
            return float("-inf")
