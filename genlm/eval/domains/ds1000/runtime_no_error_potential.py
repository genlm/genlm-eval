from typing import List
import ast, subprocess, sys, textwrap, os
from genlm.control import Potential
from genlm.control.constant import EOS, EndOfSequence

class DS1000RuntimeNoErrorPotential(Potential):
    """
    DS-1000 expensive potential: execute the harness on a complete prefix.
    Return 0.0 if no error, -inf otherwise.
    """

    def __init__(
        self,
        vocabulary: List[bytes],
        code_context: str,
        eos: EndOfSequence | None = None,
        timeout_seconds: float = 3.0,
    ):
        super().__init__(vocabulary=vocabulary, eos=eos or EOS)
        self.timeout_seconds = float(timeout_seconds)
        self.code_context = code_context

    async def prefix(self, context: List[bytes]) -> float:
        code = self._bytes_to_str(context)
        code = self._truncate_to_complete_statements(code)
        return await self._score_no_error(code)

    async def complete(self, context: List[bytes]) -> float:
        return await self.prefix(context)

    def _bytes_to_str(self, toks: List[bytes]) -> str:
        return b"".join(toks).decode("utf-8", errors="ignore") if toks else ""

    def _is_complete_python(self, code: str) -> bool:
        try:
            ast.parse(code, mode="exec")
            return True
        except SyntaxError:
            return False

    def _truncate_to_complete_statements(self, code: str) -> str:
        s = code
        if not s.strip():
            return ""
        if self._is_complete_python(s):
            return s
        while True:
            nl = s.rfind("\n")
            if nl <= 0:
                return ""
            s = s[:nl].rstrip()
            if self._is_complete_python(s):
                return s

    async def _score_no_error(self, truncated_code: str) -> float:
        if truncated_code.strip() == "":
            return 0.0
        OK, BAD = "<<<OK>>>", "<<<BAD>>>"
        script = textwrap.dedent(f"""
        import sys, traceback, warnings
        warnings.filterwarnings("ignore")
        code_context = {self.code_context!r}
        solution = {truncated_code!r}
        OK, BAD = "<<<OK>>>", "<<<BAD>>>"
        try:
            g = {{}}
            exec(code_context, g, g)  # defines test_execution
            te = g.get("test_execution")
            try:
                te(solution)
                print(OK)
            except (AssertionError, KeyError):
                # Treat harness correctness checks & missing `result` as non-fatal
                print(OK)
            except BaseException as e:
                print(BAD, type(e).__name__, str(e))
        except BaseException as e:
            print(BAD, type(e).__name__, str(e))
        """).strip()

        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                check=False, capture_output=True, text=True,
                timeout=self.timeout_seconds,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired:
            return float("-inf")

        out = (proc.stdout or "") + (proc.stderr or "")
        ok = any(line.strip() == OK for line in out.splitlines())
        bad = any(line.strip().startswith(BAD) for line in out.splitlines())
        return 0.0 if ok and not bad else float("-inf")
