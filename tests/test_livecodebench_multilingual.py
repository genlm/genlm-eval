import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fixtures.lcb_solutions import SOLUTIONS, WRONG
from fixtures.mlcb_solutions import (
    SOLUTIONS as MLCB_SOLUTIONS,
    SUM_N_INPUTS,
    SUM_N_OUTPUTS,
)

from genlm.eval.domains.livecodebench.harness import passed_all
from genlm.eval.domains.livecodebench_multilingual import (
    LANGUAGES,
    LocalSubprocessExecutor,
    MultilingualLCBDataset,
    MultilingualLCBEvaluator,
    MultilingualLCBInstance,
    extract_code,
    format_multilingual_prompt,
    is_toolchain_available,
    multilingual_chat_messages,
    resolve_language,
)
from genlm.eval.domains.livecodebench_multilingual.executor import _TOOLCHAIN
from genlm.eval.domains.livecodebench_multilingual.vendored import testing_plang
from genlm.eval.domains.livecodebench_multilingual.vendored.testing_plang import (
    SubprocessConfig,
)

FIXTURE = str(pathlib.Path(__file__).parent / "fixtures" / "lcb_sample.jsonl")
# Fixture stdin problems (functional/LeetCode rows are filtered out).
STDIN_QIDS = ["abc333_a", "abc387_a"]


# ---- language registry ----


def test_registry_has_16_languages():
    assert len(LANGUAGES) == 16
    assert sum(1 for v in LANGUAGES.values() if v.source == "multilcb") == 11
    assert sum(1 for v in LANGUAGES.values() if v.source == "agnostics") == 5


@pytest.mark.parametrize(
    "name,key",
    [
        ("cpp", "c++"),
        ("C++", "c++"),
        ("csharp", "c#"),
        ("JS", "javascript"),
        ("ts", "typescript"),
        ("python", "python"),
        ("OCaml", "ocaml"),
    ],
)
def test_resolve_language_aliases(name, key):
    assert resolve_language(name).key == key


def test_resolve_unknown_language_raises():
    with pytest.raises(ValueError, match="unknown language"):
        resolve_language("cobol")


# ---- dataset ----


def test_dataset_is_stdin_only_with_composite_id():
    ds = MultilingualLCBDataset.from_jsonl(FIXTURE, "c++")
    insts = list(ds)
    assert sorted(i.question_id for i in insts) == sorted(STDIN_QIDS)
    for i in insts:
        assert i.testtype == "stdin"
        assert i.language == "c++"
        assert i.instance_id == f"{i.question_id}@c++"
        assert isinstance(i, MultilingualLCBInstance)


def test_dataset_validates_language():
    with pytest.raises(ValueError, match="unknown language"):
        MultilingualLCBDataset.from_jsonl(FIXTURE, "cobol")


# ---- prompt byte-parity with Multi-LCB ----

# Multi-LCB literal prompt constants (lcb_runner/prompts/code_generation.py).
_MULTILCB_SYS = (
    "You are an expert {Plang} programmer. You will be given a question (problem "
    "specification) and will generate a correct {Plang} program that matches the "
    "specification and passes all tests."
)
_MULTILCB_FMT = (
    "Read the inputs from stdin solve the problem and write the answer to stdout (do not "
    "directly test on the sample inputs). Enclose your code within delimiters as follows. "
    "Ensure that when the {plang} program runs, it reads the inputs, runs the algorithm "
    "and writes output to STDOUT.\n\n"
)
_PLANG_DISPLAY = {
    "c++": "C++",
    "c#": "C#",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "python": "Python",
    "java": "Java",
    "rust": "Rust",
    "go": "Go",
    "ruby": "Ruby",
    "php": "php",
    "kotlin": "Kotlin",
}
_PLANG_FENCE = {"c++": "cpp", "c#": "csharp"}  # others: fence name == key


def _multilcb_reference(qc, plang):
    fence = _PLANG_FENCE.get(plang, plang)
    comment = "#" if plang in ("python", "ruby") else "//"
    sysm = _MULTILCB_SYS.format(Plang=_PLANG_DISPLAY[plang])
    user = f"### Question:\n{qc}\n\n"
    user += f"### Format: {_MULTILCB_FMT.format(plang=fence)}\n"
    user += f"```{fence}\n{comment} YOUR CODE HERE\n```\n\n"
    user += "### Answer: (use the provided format with backticks)\n\n"
    return sysm, user


def test_prompt_matches_multilcb_structure_per_language():
    ds = MultilingualLCBDataset.from_jsonl(FIXTURE, "c++")
    inst = next(iter(ds))
    sys_msg, user_msg = multilingual_chat_messages(inst)
    assert sys_msg["content"].startswith("You are an expert C++ programmer.")
    assert "### Question:" in user_msg["content"]
    assert "### Format:" in user_msg["content"]
    assert "```cpp\n// YOUR CODE HERE\n```" in user_msg["content"]
    assert (
        user_msg["content"]
        .rstrip()
        .endswith("### Answer: (use the provided format with backticks)")
    )


@pytest.mark.parametrize("plang", list(_PLANG_DISPLAY))
def test_prompt_byte_identical_to_multilcb(plang):
    qc = "Given an integer n, print 2n."
    inst = MultilingualLCBInstance(
        instance_id=f"x@{plang}",
        question_id="x",
        language=plang,
        question_content=qc,
        eval_sample={},
    )
    sysm, user = (m["content"] for m in multilingual_chat_messages(inst))
    ref_sys, ref_user = _multilcb_reference(qc, plang)
    assert sysm == ref_sys
    assert user == ref_user


def test_extractor_matches_multilcb():
    # first block (Multi-LCB picks result[0], not the last)
    assert (
        extract_code("```python\nprint('a')\n```\n```python\nprint('b')\n```").strip()
        == "print('a')"
    )
    assert "YOUR CODE HERE" not in extract_code(
        "```python\n# YOUR CODE HERE\nprint(1)\n```"
    )
    assert (
        extract_code(
            "<think>```python\nbad()\n```</think>\n```python\ngood()\n```"
        ).strip()
        == "good()"
    )
    assert extract_code("no fenced code") == ""


# ---- prompt nudges and byte-parity cross-check ----

_NUDGE_SUBSTR = {
    "lua": "Lua 5.1",
    "julia": "Julia 1.11",
    "r": "readLines",
    "ocaml": "Scanf",
    "fortran": "implicit none",
}


@pytest.mark.parametrize("lang,substr", list(_NUDGE_SUBSTR.items()))
def test_low_resource_nudges(lang, substr):
    inst = MultilingualLCBInstance(
        instance_id=f"x@{lang}",
        question_id="x",
        language=lang,
        question_content="q",
        eval_sample={},
    )
    assert substr in multilingual_chat_messages(inst)[0]["content"]


@pytest.mark.parametrize("lang", ["python", "c++", "java"])
def test_multilcb_language_has_no_nudge(lang):
    inst = MultilingualLCBInstance(
        instance_id=f"x@{lang}",
        question_id="x",
        language=lang,
        question_content="q",
        eval_sample={},
    )
    assert multilingual_chat_messages(inst)[0]["content"] == _MULTILCB_SYS.format(
        Plang=_PLANG_DISPLAY[lang]
    )


def test_byte_parity_maps_match_registry():
    for plang, disp in _PLANG_DISPLAY.items():
        assert LANGUAGES[plang].display == disp
        assert LANGUAGES[plang].md_fence == _PLANG_FENCE.get(plang, plang)


# ---- format_multilingual_prompt (token-id entry point) ----


class _FakeTok:
    def __init__(self):
        self.apply_kw = None
        self.encode_kw = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kw):
        self.apply_kw = {"add_generation_prompt": add_generation_prompt, **kw}
        return "CHAT:" + messages[0]["content"] + "||" + messages[1]["content"]

    def encode(self, text, add_special_tokens=True):
        self.encode_kw = {"text": text, "add_special_tokens": add_special_tokens}
        return [len(text)]


def test_format_multilingual_prompt_raw_completion():
    tok = _FakeTok()
    inst = _instance("python", ["1\n"], ["1\n"])
    sysm, user = (m["content"] for m in multilingual_chat_messages(inst))
    out = format_multilingual_prompt(tok, inst, use_chat_format=False)
    assert out == [len(f"{sysm}\n\n{user}")]
    assert tok.encode_kw["text"] == f"{sysm}\n\n{user}"
    assert tok.encode_kw["add_special_tokens"] is True


def test_format_multilingual_prompt_chat_and_thinking():
    tok = _FakeTok()
    inst = _instance("python", ["1\n"], ["1\n"])
    format_multilingual_prompt(tok, inst, use_chat_format=True, enable_thinking=True)
    assert tok.apply_kw["add_generation_prompt"] is True
    assert tok.apply_kw["enable_thinking"] is True
    assert tok.encode_kw["add_special_tokens"] is False


# ---- executor / evaluator helpers ----


def _instance(language, inputs, outputs, qid="sum"):
    return MultilingualLCBInstance(
        instance_id=f"{qid}@{language}",
        question_id=qid,
        language=language,
        question_content="add two integers on one line",
        eval_sample={
            "input_output": json.dumps(
                {"inputs": inputs, "outputs": outputs, "fn_name": None}
            )
        },
    )


# ---- malformed / edge-case grading ----


def test_malformed_eval_sample_scores_failure_without_crashing():
    ev = MultilingualLCBEvaluator()
    for bad in (
        {"input_output": ""},
        {"input_output": "{}"},
        {"input_output": "not json"},
    ):
        inst = MultilingualLCBInstance(
            instance_id="b@python",
            question_id="b",
            language="python",
            question_content="q",
            eval_sample=bad,
        )
        res = ev.evaluate_sample(inst, "```python\nprint(1)\n```")
        assert res.score == 0.0 and res.desc == "malformed eval_sample"


def test_evaluator_early_return_descs():
    ev = MultilingualLCBEvaluator()
    assert ev.evaluate_sample(
        _instance("python", ["1\n"], ["1\n"]), "no code"
    ).desc == ("empty code")
    missing = MultilingualLCBInstance(
        instance_id="x@python",
        question_id="x",
        language="python",
        question_content="q",
        eval_sample={},
    )
    r = ev.evaluate_sample(missing, "```python\nprint(1)\n```")
    assert r.score == 0.0 and r.desc == "missing eval_sample"
    empty = _instance("python", [], [])
    r2 = ev.evaluate_sample(empty, "```python\nprint(1)\n```")
    assert r2.score == 0.0 and r2.desc == "no test inputs"


def test_executor_metadata_shape():
    ex = LocalSubprocessExecutor()
    good, gm = ex.run(
        "a,b=map(int,input().split())\nprint(a+b)", ["2 3\n"], ["5\n"], "python", 6
    )
    assert good is True and gm["per_test"] == ["PASSED"] and gm["n_tests"] == 1
    wrong, wm = ex.run("print(0)", ["2 3\n"], ["5\n"], "python", 6)
    assert wrong is False and wm["per_test"] == ["FAILED"]


def test_timeout_grades_failure():
    solved, _ = LocalSubprocessExecutor().run(
        "while True:\n    pass", ["1\n"], ["1\n"], "python", 1
    )
    assert solved is False


def test_runtime_error_nonzero_exit_fails():
    solved, meta = LocalSubprocessExecutor().run(
        "raise ValueError('boom')", ["2 3\n"], ["5\n"], "python", 6
    )
    assert solved is False and meta["per_test"] == ["EXECFAIL"]


@pytest.mark.parametrize(
    "keyword", ["ValueError", "SyntaxError", "out of memory", "TimeoutExpired"]
)
def test_correct_program_with_stderr_keyword_still_passes(keyword):
    # Exit-code-first status fix: a correct program writing an error keyword to stderr passes.
    ev = MultilingualLCBEvaluator(timeout_seconds=10.0)
    inst = _instance("python", ["2 3\n"], ["5\n"])
    gen = (
        "```python\nimport sys\n"
        f"sys.stderr.write('{keyword}: not really\\n')\n"
        "a,b=map(int,input().split())\nprint(a+b)\n```"
    )
    assert ev.evaluate_sample(inst, gen).score == 1.0


def test_executor_rejects_unwired_language():
    with pytest.raises(NotImplementedError, match="not yet wired"):
        LocalSubprocessExecutor().run("x", ["1\n"], ["1\n"], "brainfuck", 6)


def test_prepare_raises_for_missing_toolchain():
    missing = next(
        (lang for lang in MLCB_SOLUTIONS if not is_toolchain_available(lang)), None
    )
    if missing is None:
        pytest.skip("all toolchains installed")
    with pytest.raises(RuntimeError, match="toolchain for"):
        LocalSubprocessExecutor().prepare(missing)


def test_all_16_languages_wired_in_executor():
    for lang in LANGUAGES:
        assert lang in testing_plang.eval_scripts, f"{lang} not wired"


# ---- grading semantics ----


def test_exact_grading_is_stricter_than_lenient():
    # A program printing "true" against expected "True": lenient aliases (pass), exact does not.
    inst = _instance("python", ["x\n"], ["True\n"])
    code = "```python\ninput()\nprint('true')\n```"
    assert (
        MultilingualLCBEvaluator(grading="lenient").evaluate_sample(inst, code).score
        == 1.0
    )
    assert (
        MultilingualLCBEvaluator(grading="exact").evaluate_sample(inst, code).score
        == 0.0
    )
    good = "```python\ninput()\nprint('True')\n```"
    assert (
        MultilingualLCBEvaluator(grading="lenient").evaluate_sample(inst, good).score
        == 1.0
    )
    assert (
        MultilingualLCBEvaluator(grading="exact").evaluate_sample(inst, good).score
        == 1.0
    )


def test_invalid_grading_raises():
    with pytest.raises(ValueError, match="grading must be"):
        LocalSubprocessExecutor(grading="fuzzy")


def test_toolchain_table_covers_registry():
    assert set(LANGUAGES) <= set(_TOOLCHAIN)
    for lang in LANGUAGES:
        fn, ext = testing_plang.eval_scripts[lang]
        assert callable(fn) and isinstance(ext, str) and ext.startswith(".")


# ---- evaluator caching and prepare-once ----


class _CountingExecutor:
    def __init__(self):
        self.prepared = []
        self.runs = 0

    def prepare(self, language):
        if language not in self.prepared:
            self.prepared.append(language)

    def run(self, code, inputs, outputs, language, timeout):
        self.runs += 1
        return True, {"status": "ok", "per_test": ["PASSED"], "n_tests": len(outputs)}


def test_evaluator_caches_and_prepares_once():
    ex = _CountingExecutor()
    ev = MultilingualLCBEvaluator(executor=ex)
    inst = _instance("python", ["1\n"], ["1\n"])
    gen = "```python\nprint(1)\n```"
    ev.evaluate_sample(inst, gen)
    ev.evaluate_sample(inst, gen)  # cache hit
    assert ex.runs == 1
    ev.evaluate_sample(inst, "```python\nprint(2)\n```")
    assert ex.runs == 2
    assert ex.prepared == ["python"]


# ---- multi-test parity with the existing LCB harness ----


def test_parity_multi_test_and_partial():
    inputs = ["2 3\n", "10 20\n", "0 0\n"]
    outputs = ["5 \n", "30\n", "0\n"]  # trailing space exercises normalization
    sample = {
        "input_output": json.dumps(
            {"inputs": inputs, "outputs": outputs, "fn_name": None}
        )
    }
    inst = _instance("python", inputs, outputs)
    ev = MultilingualLCBEvaluator(timeout_seconds=6.0)
    good = "```python\na,b=map(int,input().split())\nprint(a+b)\n```"
    partial = "```python\na,b=map(int,input().split())\nprint(a+b if a else 999)\n```"
    for gen in (good, partial):
        ours = ev.evaluate_sample(inst, gen).score == 1.0
        assert ours == passed_all(sample, extract_code(gen))
    assert ev.evaluate_sample(inst, good).score == 1.0
    assert ev.evaluate_sample(inst, partial).score == 0.0


def test_python_grading_matches_existing_harness():
    # Multilingual python path agrees with the default-LCB harness (passed_all) on real fixtures.
    # The lenient comparator diverges from default-LCB on bool-alias / float-tolerance outputs;
    # test_python_lenient_more_permissive_than_default pins that intentional gap.
    ds = {
        i.question_id: i for i in MultilingualLCBDataset.from_jsonl(FIXTURE, "python")
    }
    with open(FIXTURE) as f:
        rows = {
            json.loads(line)["question_id"]: json.loads(line)
            for line in f
            if line.strip()
        }
    ev = MultilingualLCBEvaluator(timeout_seconds=6.0)
    for qid in STDIN_QIDS:
        for gen in (SOLUTIONS[qid], WRONG):
            ours = ev.evaluate_sample(ds[qid], gen).score == 1.0
            theirs = passed_all(rows[qid]["eval_sample"], extract_code(gen))
            assert ours == theirs, f"{qid}: ours={ours} passed_all={theirs}"


# ---- python lenient vs default-LCB divergence ----


def _ours_solves(printed, expected):
    """Grade a python program via our lenient path."""
    code = f"import sys\nsys.stdout.write({printed!r})\n"
    solved, _ = LocalSubprocessExecutor().run(code, ["x\n"], [expected], "python", 5.0)
    return solved


def _default_solves(printed, expected):
    """Grade the same program via the default-LCB harness."""
    code = f"import sys\nsys.stdout.write({printed!r})\n"
    es = {
        "input_output": json.dumps(
            {"inputs": ["x\n"], "outputs": [expected], "fn_name": None}
        )
    }
    return passed_all(es, code)


# Cases where lenient accepts but the default-LCB grader rejects.
_DIVERGE = [
    ("true\n", "True\n"),
    ("false\n", "False\n"),
    ("1.000001\n", "1.000002\n"),  # within abs_tol=1e-5
]


@pytest.mark.parametrize("printed,expected", _DIVERGE)
def test_python_lenient_more_permissive_than_default(printed, expected):
    assert _ours_solves(printed, expected) is True
    assert _default_solves(printed, expected) is False


_AGREE_PASS = [("6\n", "6\n"), ("6 \n", "6\n"), ("5\n", "5.0\n")]
_AGREE_FAIL = [("1.0001\n", "1.0002\n"), ("6\n7\n", "6\n")]


@pytest.mark.parametrize("printed,expected", _AGREE_PASS)
def test_python_graders_agree_on_pass(printed, expected):
    assert _ours_solves(printed, expected) is True
    assert _default_solves(printed, expected) is True


@pytest.mark.parametrize("printed,expected", _AGREE_FAIL)
def test_python_graders_agree_on_fail(printed, expected):
    assert _ours_solves(printed, expected) is False
    assert _default_solves(printed, expected) is False


def test_status_str_is_name_stable():
    # executor.py builds metadata["status"] from str(meta.error); must equal enum member name.
    for member in testing_plang.Status:
        assert str(member) == member.name
    for member in testing_plang.TestScore:
        assert str(member) == member.name


# ---- per-language toolchain smoke test ----

# (binary, fenced generation) per language for a simple sum-two-ints problem.
SUM_IO = (["2 3\n", "10 20\n"], ["5\n", "30\n"])
SUM_SOLUTIONS = {
    "python": ("python", "```python\na,b=map(int,input().split())\nprint(a+b)\n```"),
    "c++": (
        "g++",
        '```cpp\n#include <iostream>\nint main(){long long a,b;std::cin>>a>>b;std::cout<<a+b<<"\\n";return 0;}\n```',
    ),
    "rust": (
        "rustc",
        '```rust\nuse std::io::*;\nfn main(){let mut s=String::new();stdin().read_line(&mut s).unwrap();let v:Vec<i64>=s.trim().split_whitespace().map(|x|x.parse().unwrap()).collect();println!("{}",v[0]+v[1]);}\n```',
    ),
    "javascript": (
        "node",
        "```javascript\nconst l=require('fs').readFileSync(0,'utf8').trim().split(/\\s+/).map(Number);console.log(l[0]+l[1]);\n```",
    ),
    "lua": ("luajit", '```lua\nlocal a,b=io.read("*n","*n")\nprint(a+b)\n```'),
    "julia": (
        "julia",
        "```julia\na,b=split(readline())\nprintln(parse(Int,a)+parse(Int,b))\n```",
    ),
    "r": (
        "Rscript",
        '```r\ninput<-readLines(file("stdin"))\nv<-as.integer(strsplit(input[1]," ")[[1]])\ncat(v[1]+v[2]); cat("\\n")\n```',
    ),
    "ocaml": (
        "ocaml",
        '```ocaml\nlet () = Scanf.scanf " %d %d" (fun a b -> Printf.printf "%d\\n" (a+b))\n```',
    ),
    "fortran": (
        "gfortran",
        "```fortran\nprogram main\nimplicit none\ninteger :: a,b\nread(*,*) a,b\nprint *, a+b\nend program main\n```",
    ),
    "java": (
        "javac",
        "```java\nimport java.util.*;\npublic class Main{public static void main(String[] x){Scanner s=new Scanner(System.in);long a=s.nextLong(),b=s.nextLong();System.out.println(a+b);}}\n```",
    ),
    "c#": (
        "mcs",
        "```csharp\nusing System;\nclass Program{static void Main(){var p=Console.ReadLine().Split(' ');Console.WriteLine(long.Parse(p[0])+long.Parse(p[1]));}}\n```",
    ),
    "go": (
        "go",
        '```go\npackage main\nimport("bufio";"fmt";"os")\nfunc main(){r:=bufio.NewReader(os.Stdin);var a,b int64;fmt.Fscan(r,&a,&b);fmt.Println(a+b)}\n```',
    ),
    "ruby": ("ruby", "```ruby\na,b=gets.split.map(&:to_i)\nputs a+b\n```"),
    "php": (
        "php",
        "```php\n<?php\nlist($a,$b)=array_map('intval',explode(' ',trim(fgets(STDIN))));\necho $a+$b,\"\\n\";\n```",
    ),
    "kotlin": (
        "kotlinc",
        '```kotlin\nfun main(){val (a,b)=readLine()!!.trim().split(" ").map{it.toLong()};println(a+b)}\n```',
    ),
    "typescript": (
        "deno",
        "```typescript\nconst data = await new Response(Deno.stdin.readable).text();\nconst [a,b]=data.trim().split(/\\s+/).map(Number);\nconsole.log(a+b);\n```",
    ),
}


@pytest.mark.parametrize("language", list(SUM_SOLUTIONS))
def test_executor_grades_good_and_wrong(language):
    tool, good = SUM_SOLUTIONS[language]
    if shutil.which(tool) is None:
        pytest.skip(f"{tool} not installed")
    ev = MultilingualLCBEvaluator(timeout_seconds=20.0)
    inst = _instance(language, *SUM_IO)
    assert ev.evaluate_sample(inst, good).score == 1.0
    assert ev.evaluate_sample(inst, "no code here").score == 0.0


# ---- per-language verdict matrix ----

# Contract: (expect_solved, allowed status strings, expected per_test list).
CONTRACT = {
    "correct": (True, {"ok"}, ["PASSED", "PASSED", "PASSED"]),
    "wrong_output": (False, {"WrongAnswer"}, ["FAILED"]),
    "partial": (False, {"WrongAnswer"}, ["PASSED", "PASSED", "FAILED"]),
    "runtime_error": (False, {"Exception", "AbnormalTermination"}, ["EXECFAIL"]),
    "compile_error": (
        False,
        {"SyntaxError", "BuildFailed", "Exception", "AbnormalTermination"},
        ["EXECFAIL"],
    ),
}

_MATRIX_CASES = [
    (lang, variant) for lang, vs in MLCB_SOLUTIONS.items() for variant in vs
]


@pytest.mark.parametrize("language,variant", _MATRIX_CASES)
def test_verdict_matrix(language, variant):
    if not is_toolchain_available(language):
        pytest.skip(f"{language} toolchain not installed")
    expect_solved, allowed_status, expect_per_test = CONTRACT[variant]
    ex = LocalSubprocessExecutor()
    ex.prepare(language)
    solved, meta = ex.run(
        MLCB_SOLUTIONS[language][variant],
        SUM_N_INPUTS,
        SUM_N_OUTPUTS,
        language,
        timeout=20.0,
    )
    assert solved is expect_solved, f"{language}/{variant}: solved={solved} meta={meta}"
    assert meta["per_test"] == expect_per_test, f"{language}/{variant}: {meta}"
    assert meta["status"] in allowed_status, (
        f"{language}/{variant}: status={meta['status']}"
    )
    assert meta["n_tests"] == len(SUM_N_OUTPUTS)


@pytest.mark.parametrize("language", ["python", "c++"])
def test_evaluator_end_to_end_fenced(language):
    # Drives the full evaluator path (extract_code from fenced generation, then grade).
    if not is_toolchain_available(language):
        pytest.skip("toolchain absent")
    fence = resolve_language(language).md_fence
    inst = MultilingualLCBInstance(
        instance_id=f"sum@{language}",
        question_id="sum",
        language=language,
        question_content="sum of N integers",
        eval_sample={
            "input_output": json.dumps(
                {"inputs": SUM_N_INPUTS, "outputs": SUM_N_OUTPUTS, "fn_name": None}
            )
        },
    )
    ev = MultilingualLCBEvaluator(timeout_seconds=20.0)
    good = f"```{fence}\n{MLCB_SOLUTIONS[language]['correct']}\n```"
    wrong = f"```{fence}\n{MLCB_SOLUTIONS[language]['wrong_output']}\n```"
    assert ev.evaluate_sample(inst, good).score == 1.0
    assert ev.evaluate_sample(inst, wrong).score == 0.0


# ---- robustness ----


def _pgrep(pattern):
    """PIDs whose command line matches pattern (empty list if none)."""
    found = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return [p for p in found.stdout.split() if p]


def test_timeout_is_bounded_and_classified():
    ex = LocalSubprocessExecutor()
    t0 = time.time()
    solved, meta = ex.run("while True:\n    pass\n", ["x\n"], ["x\n"], "python", 1.0)
    elapsed = time.time() - t0
    assert solved is False
    assert meta["status"] == "TimeoutExpired"
    assert meta["per_test"] == ["EXECFAIL"]
    assert elapsed < 15.0, f"timeout took {elapsed:.1f}s; group kill may be leaking"


def test_large_output_does_not_hang():
    ex = LocalSubprocessExecutor()
    flood = "import sys\nsys.stdout.write('z' * 8_000_000)\n"
    t0 = time.time()
    solved, meta = ex.run(flood, ["3\n1 2 3\n"], ["6\n"], "python", 5.0)
    elapsed = time.time() - t0
    assert solved is False
    assert meta["per_test"] == ["FAILED"]
    assert elapsed < 15.0, f"large-output grading took {elapsed:.1f}s"


def test_grandchild_reaped_after_timeout():
    sentinel = "mlcb_robustness_sentinel_b91e"
    prog = (
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)', '{sentinel}'])\n"
        "while True:\n    time.sleep(0.05)\n"
    )
    ex = LocalSubprocessExecutor()
    solved, _ = ex.run(prog, ["x\n"], ["x\n"], "python", 1.0)
    assert solved is False
    survivors = _pgrep(sentinel)
    deadline = time.time() + 5.0
    while survivors and time.time() < deadline:
        time.sleep(0.1)
        survivors = _pgrep(sentinel)
    for pid in survivors:
        subprocess.run(["kill", "-9", pid], capture_output=True)
    assert survivors == [], f"leaked child processes: {survivors}"


@pytest.mark.parametrize(
    "language,capped",
    [
        ("python", True),
        ("c++", True),
        ("go", True),
        ("java", True),
        ("rust", False),
        ("javascript", False),
        ("typescript", False),
        ("julia", False),
    ],
)
def test_memory_limit_config(language, capped):
    # Native/GC runtimes reserve huge virtual address space; they are exempt from RLIMIT_AS.
    assert SubprocessConfig(plang=language).limit_memory is capped


def test_prepare_julia_warms_once(monkeypatch):
    from genlm.eval.domains.livecodebench_multilingual import executor as executor_mod

    calls = []
    monkeypatch.setattr(executor_mod, "is_toolchain_available", lambda lang: True)
    monkeypatch.setattr(
        executor_mod.subprocess, "run", lambda *a, **k: calls.append(a[0])
    )
    ex = LocalSubprocessExecutor()
    ex.prepare("julia")
    ex.prepare("julia")
    assert len(calls) == 1, f"expected one warmup call, got {calls}"
    assert "julia" in calls[0][0]


@pytest.mark.parametrize("language", ["go", "python"])
def test_prepare_makes_no_subprocess_call(monkeypatch, language):
    from genlm.eval.domains.livecodebench_multilingual import executor as executor_mod

    calls = []
    monkeypatch.setattr(executor_mod, "is_toolchain_available", lambda lang: True)
    monkeypatch.setattr(executor_mod.subprocess, "run", lambda *a, **k: calls.append(a))
    LocalSubprocessExecutor().prepare(language)
    assert calls == []


def test_prepare_missing_toolchain_raises_and_not_marked(monkeypatch):
    from genlm.eval.domains.livecodebench_multilingual import executor as executor_mod

    monkeypatch.setattr(executor_mod, "is_toolchain_available", lambda lang: False)
    ex = LocalSubprocessExecutor()
    with pytest.raises(RuntimeError, match="toolchain"):
        ex.prepare("c++")
    assert "c++" not in ex._prepared


def test_unicode_io_round_trips():
    ex = LocalSubprocessExecutor()
    echo = "import sys\nsys.stdout.write(sys.stdin.read())\n"
    payload = ["héllo wörld ☃\n"]
    solved, meta = ex.run(echo, payload, payload, "python", 5.0)
    assert solved is True and meta["per_test"] == ["PASSED"]
    bad, _ = ex.run(echo, payload, ["different\n"], "python", 5.0)
    assert bad is False


def test_concurrent_grading_no_crosstalk():
    correct = "import sys\nprint(sum(int(x) for x in sys.stdin.read().split()[1:]))\n"
    wrong = "import sys\nprint(sum(int(x) for x in sys.stdin.read().split()[1:]) + 1)\n"
    jobs = [(correct, True), (wrong, False)] * 6

    def grade(job):
        code, _ = job
        ex = LocalSubprocessExecutor()
        solved, _ = ex.run(code, ["3\n1 2 3\n"], ["6\n"], "python", 10.0)
        return solved

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(grade, jobs))
    assert results == [expected for _, expected in jobs]


# ---- differential: agreement with upstream Multi-LCB executor ----

_MULTI_LCB = Path(__file__).resolve().parents[1].parent / "Multi-LCB"


def _load_upstream():
    """Import upstream testing_plang as a standalone module, or None if unavailable."""
    path = _MULTI_LCB / "lcb_runner" / "evaluation" / "testing_plang.py"
    if not path.exists():
        return None
    root = str(_MULTI_LCB)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        spec = importlib.util.spec_from_file_location("upstream_testing_plang", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:  # noqa: BLE001
        return None


_UPSTREAM = _load_upstream()
_SHARED = sorted(set(MLCB_SOLUTIONS) & set(_UPSTREAM.eval_scripts)) if _UPSTREAM else []
_DIFF_CASES = [(lang, v) for lang in _SHARED for v in MLCB_SOLUTIONS[lang]]


@pytest.mark.skipif(not _MULTI_LCB.exists(), reason="../Multi-LCB not present")
def test_upstream_actually_loaded():
    # Guard against silent self-disable: if the repo is present, import must succeed.
    assert _UPSTREAM is not None, (
        "../Multi-LCB present but testing_plang failed to import"
    )
    assert _DIFF_CASES, "upstream loaded but no shared languages resolved"


@pytest.mark.differential
@pytest.mark.skipif(_UPSTREAM is None, reason="../Multi-LCB not importable")
@pytest.mark.parametrize("language,variant", _DIFF_CASES)
def test_verdict_agrees_with_upstream(language, variant):
    if not is_toolchain_available(language):
        pytest.skip(f"{language} toolchain absent")
    code = MLCB_SOLUTIONS[language][variant]
    ex = LocalSubprocessExecutor()
    ex.prepare(language)
    ours, _ = ex.run(code, SUM_N_INPUTS, SUM_N_OUTPUTS, language, 20.0)
    scores, _ = _UPSTREAM.eval_plang_code(
        code, list(SUM_N_INPUTS), list(SUM_N_OUTPUTS), language, 20
    )
    theirs = bool(scores) and all(s.value > 0 for s in scores)
    assert ours == theirs, f"{language}/{variant}: ours={ours} upstream={theirs}"
