import logging
import os
import random
import subprocess
import sys
import ast
import textwrap
import tempfile
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Type

from datasets import load_dataset

from genlm.eval.domains.ds1000.utils import _sandbox_env, _postprocess_code
from genlm.eval.core import EvaluationResult, Instance, Dataset, Evaluator

log = logging.getLogger(__name__)

################
# DS-1000 Data #
################

class DS1000Instance(Instance):
    """Schema for a DS-1000 instance."""

    prompt: str 
    code_context: str 
    metadata: Dict[str, Any]
    reference_code: Optional[str] = None


class DS1000Dataset(Dataset[DS1000Instance]):
    """Dataset for DS-1000 evaluation (Lai et al., 2023)."""

    def __init__(self, rows: List[Mapping[str, Any]]):
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[DS1000Instance]:
        for i, row in enumerate(self._rows):
            yield DS1000Instance(
                prompt=str(row.get("prompt", "")),
                code_context=str(row.get("code_context", "")).strip(),
                reference_code=row.get("reference_code"),
                metadata=(row.get("metadata") or {}),
                instance_id=i,
            )

    @property
    def schema(self) -> Type[DS1000Instance]:
        return DS1000Instance

    @classmethod
    def from_hf(
        cls,
        split: str = "test",
        libraries: Optional[Sequence[str]] = None,
        perturbation_types: Optional[Sequence[str]] = None,
        max_instances: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 1234,
        cache_dir: Optional[str] = None,
    ) -> "DS1000Dataset":
        """Load and (optionally) filter DS-1000 from Hugging Face."""
        ds = load_dataset("xlangai/DS-1000", split=split, cache_dir=cache_dir)
        rows: List[Mapping[str, Any]] = list(ds)

        lib_set = {x.lower() for x in libraries} if libraries else None
        pt_set = {x.lower() for x in perturbation_types} if perturbation_types else None

        def _keep(r: Mapping[str, Any]) -> bool:
            m = (r.get("metadata") or {})
            lib_ok = True if not lib_set else str(m.get("library", "")).lower() in lib_set
            pt_ok = True if not pt_set else str(m.get("perturbation_type", "")).lower() in pt_set
            return lib_ok and pt_ok

        rows = [r for r in rows if _keep(r)]

        if shuffle:
            rnd = random.Random(seed)
            rnd.shuffle(rows)

        if isinstance(max_instances, int) and max_instances >= 0:
            rows = rows[:max_instances]

        log.info("Loaded DS-1000: %d instances (split=%s)", len(rows), split)
        return cls(rows)

################
#   Evaluator  #
################

class DS1000Evaluator(Evaluator[DS1000Instance]):
    def __init__(
            self,
            python_executable: Optional[str] = None,
            timeout_seconds: float = 15.0,
            extra_env: Optional[Dict[str, str]] = None,
            max_log_chars: int = 4000
        ) -> None:
        self.python_executable = python_executable or sys.executable
        self.timeout_seconds = float(timeout_seconds)
        self.extra_env = dict(extra_env or {})
        self.max_log_chars = int(max_log_chars)
        # Markers for detecting PASS/FAIL in output
        self.marker_pass = "<<<DS1000_PASS>>>"
        self.marker_fail = "<<<DS1000_FAIL>>>"

    def assigns_result(self, code: str) -> bool:
        try:
            tree = ast.parse(code)
            for n in ast.walk(tree):
                if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "result" for t in n.targets):
                    return True
                if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", None) == "result":
                    return True
        except Exception:
            pass
        return False
    
    def evaluate_sample(self, instance: DS1000Instance, response: str) -> EvaluationResult:
        solution = _postprocess_code(response)
        if not solution.strip():
            return EvaluationResult(score=0.0, desc="empty solution", metadata=instance.metadata)

        script = self._build_harness_script(instance.code_context, solution)
        ok, _, _, _ = self._run_in_subprocess(script)

        # Summarize with clear sections, trim to max_log_chars
        def _trim(s: str) -> str:
            return s if len(s) <= self.max_log_chars else (s[:self.max_log_chars] + "\n...[truncated]")

        desc = _trim(solution)
        return EvaluationResult(score=1.0 if ok else 0.0, desc=desc, metadata=instance.metadata)

    def _build_harness_script(self, code_context: str, solution: str) -> str:
        """Load test_execution() and run it with solution."""
        return textwrap.dedent(f"""
        # -*- coding: utf-8 -*-
        import sys, traceback

        code_context = {code_context!r}
        solution = {solution!r}
        g = {{"__name__": "__main__"}}
        try:
            exec(code_context, g, g)
        except BaseException as e:
            print("{self.marker_fail} HARNESS_EXEC_ERROR:", repr(e), flush=True)
            traceback.print_exc()
            sys.exit(1)

        test_execution = g.get("test_execution")
        if not callable(test_execution):
            print("{self.marker_fail} MISSING_test_execution", flush=True)
            sys.exit(1)

        try:
            _ret = test_execution(solution)
            if _ret is False:
                print("{self.marker_fail} TEST_RETURNED_FALSE", flush=True)
                sys.exit(3)
        except Exception as e:
            print("{self.marker_fail}", repr(e), flush=True)
            traceback.print_exc()
            sys.exit(2)

        # Runs test_string when the problem defines it; both checks must pass.
        test_string = g.get("test_string")
        if "test_string(" in code_context and callable(test_string):
            try:
                _rs = test_string(solution)
                if _rs is False:
                    print("{self.marker_fail} TEST_STRING_RETURNED_FALSE", flush=True)
                    sys.exit(4)
            except Exception as e:
                print("{self.marker_fail}", repr(e), flush=True)
                traceback.print_exc()
                sys.exit(2)

        print("{self.marker_pass}", flush=True)
        sys.exit(0)
        """).strip()

    def _run_in_subprocess(self, script: str) -> Tuple[bool, int, str, str]:
        with tempfile.TemporaryDirectory(prefix="ds1000_") as td:
            path = os.path.join(td, "harness.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write(script + "\n")

            # Build a sandboxed env
            env = _sandbox_env(td, extra_env={**{"MPLBACKEND": "Agg"}, **self.extra_env})
            cmd = [self.python_executable, "-B", path]

            try:
                proc = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    env=env,
                    cwd=td,
                )
            except subprocess.TimeoutExpired:
                return (False, 124, "", f"timeout after {self.timeout_seconds:.1f}s")

            out = proc.stdout or ""
            err = proc.stderr or ""
            rc = int(proc.returncode)
            pass_line = bool(self.marker_pass in out)
            fail_line = bool(self.marker_fail in out) or bool(self.marker_fail in err)
            ok = (rc == 0) and pass_line and (not fail_line)
            return (ok, rc, out.strip(), err.strip())

##########################
# Prompt formatter (LM)  #
##########################

def default_prompt_formatter(
        tokenizer,
        instance: DS1000Instance,
        use_chat_format: bool = False # conform with evaluator interface
    ) -> List[int]:
    return tokenizer.encode(instance.prompt)
