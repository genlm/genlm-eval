import json
import pathlib
import shutil

import pytest
from fixtures.lcb_solutions import SOLUTIONS, WRONG

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
    pass_at_k,
    pass_at_k_from_scores,
    resolve_language,
)
from genlm.eval.domains.livecodebench_multilingual.executor import _TOOLCHAIN

FIXTURE = str(pathlib.Path(__file__).parent / "fixtures" / "lcb_sample.jsonl")
# Fixture stdin problems (the 2 functional/LeetCode rows are filtered out).
STDIN_QIDS = ["abc333_a", "abc387_a"]


# ------------------------------ language registry ------------------------------ #


def test_registry_has_17_languages():
    assert len(LANGUAGES) == 17
    # 12 mainstream + 5 low-resource
    assert sum(1 for v in LANGUAGES.values() if v.source == "multilcb") == 12
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


# ------------------------------ dataset ------------------------------ #


def test_dataset_is_stdin_only_with_composite_id():
    ds = MultilingualLCBDataset.from_jsonl(FIXTURE, "c++")
    insts = list(ds)
    # 4 fixture rows yield 2 stdin (functional rows dropped)
    assert sorted(i.question_id for i in insts) == sorted(STDIN_QIDS)
    for i in insts:
        assert i.testtype == "stdin"
        assert i.language == "c++"
        assert i.instance_id == f"{i.question_id}@c++"  # composite, qid preserved
        assert isinstance(i, MultilingualLCBInstance)


def test_dataset_validates_language():
    with pytest.raises(ValueError, match="unknown language"):
        MultilingualLCBDataset.from_jsonl(FIXTURE, "cobol")


# ------------------------------ prompts ------------------------------ #


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


# ------------------------------ pass@k ------------------------------ #


@pytest.mark.parametrize(
    "n,c,k,expected",
    [
        (10, 0, 1, 0.0),
        (10, 10, 1, 1.0),
        (5, 1, 5, 1.0),
        (2, 1, 1, 0.5),
    ],
)
def test_pass_at_k(n, c, k, expected):
    assert pass_at_k(n, c, k) == pytest.approx(expected)


# ------------------------------ executor / evaluator ------------------------------ #


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


# A trivial stdin problem (read two ints on a line, print their sum) + a known-good
# solution per language, used to smoke-test each toolchain. Skipped if the compiler/
# interpreter is absent (no upstream tests_plangs entry for several of these).
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
    # 5 low-resource languages (Agnostics), graded by the newly-wired eval_script_*.
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
    # The remaining mainstream languages (toolchains absent here unless mlcb-tools is on PATH).
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
    "scala": (
        "scalac",
        '```scala\nobject Main{def main(args:Array[String]):Unit={val Array(a,b)=scala.io.StdIn.readLine().trim.split(" ").map(_.toLong);println(a+b)}}\n```',
    ),
}


@pytest.mark.parametrize("language", list(SUM_SOLUTIONS))
def test_executor_grades_good_and_wrong(language):
    tool, good = SUM_SOLUTIONS[language]
    if shutil.which(tool) is None:
        pytest.skip(f"{tool} not installed")
    # 20s run budget so interpreter/JVM startup (julia, kotlin) fits; compile uses its own
    # 60s build timeout.
    ev = MultilingualLCBEvaluator(timeout_seconds=20.0)
    inst = _instance(language, *SUM_IO)
    assert ev.evaluate_sample(inst, good).score == 1.0
    # no fence, so empty code, so fail
    assert ev.evaluate_sample(inst, "no code here").score == 0.0


def test_executor_rejects_unwired_language():
    # A language with no eval_scripts entry must raise (defensive; all 17 registry langs
    # are wired, so this is reached only via a bogus language string).
    with pytest.raises(NotImplementedError, match="not yet wired"):
        LocalSubprocessExecutor().run("x", ["1\n"], ["1\n"], "brainfuck", 6)


def test_prepare_raises_for_missing_toolchain():
    # Pick a wired language whose toolchain is absent here; assert a clear error (not a
    # cryptic FileNotFoundError). Skip if every toolchain happens to be installed.
    missing = next(
        (lang for lang in SUM_SOLUTIONS if not is_toolchain_available(lang)), None
    )
    if missing is None:
        pytest.skip("all toolchains installed")
    with pytest.raises(RuntimeError, match="toolchain for"):
        LocalSubprocessExecutor().prepare(missing)


def test_all_17_languages_wired_in_executor():
    from genlm.eval.domains.livecodebench_multilingual.vendored import testing_plang

    for lang in LANGUAGES:
        assert lang in testing_plang.eval_scripts, f"{lang} not wired"


# ------------------------------ parity ------------------------------ #


def test_python_grading_matches_existing_harness():
    # Our multilingual python path agrees with the established Python harness
    # (passed_all / vendored testing_util) on real reference solutions and a wrong one. This is
    # agreement on these problems, NOT in general: the lenient comparator diverges from default-LCB
    # on bool-alias / float-tolerance outputs. test_mlcb_consistency pins that intentional gap.
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
            # same (our) extracted code, graded by the other harness
            theirs = passed_all(rows[qid]["eval_sample"], extract_code(gen))
            assert ours == theirs, f"{qid}: ours={ours} passed_all={theirs}"


# ------------------------------ Multi-LCB byte parity ------------------------------ #

# Multi-LCB's literal prompt constants (lcb_runner/prompts/code_generation.py) + PLang maps,
# reconstructed here so we assert byte-identity rather than mere substring presence.
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
    "scala": "Scala",
    "kotlin": "Kotlin",
}
_PLANG_FENCE = {"c++": "cpp", "c#": "csharp"}  # rest of the fence names == key


def _multilcb_reference(qc, plang):
    fence = _PLANG_FENCE.get(plang, plang)
    comment = "#" if plang in ("python", "ruby") else "//"
    sysm = _MULTILCB_SYS.format(Plang=_PLANG_DISPLAY[plang])
    user = f"### Question:\n{qc}\n\n"
    user += f"### Format: {_MULTILCB_FMT.format(plang=fence)}\n"
    user += f"```{fence}\n{comment} YOUR CODE HERE\n```\n\n"
    user += "### Answer: (use the provided format with backticks)\n\n"
    return sysm, user


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
    assert user == ref_user  # byte-for-byte, incl. the 3-newline gap before the fence


def test_extractor_matches_multilcb():
    # first block (Multi-LCB picks result[0]), not the last
    assert (
        extract_code("```python\nprint('a')\n```\n```python\nprint('b')\n```").strip()
        == "print('a')"
    )
    # placeholder removed
    assert "YOUR CODE HERE" not in extract_code(
        "```python\n# YOUR CODE HERE\nprint(1)\n```"
    )
    # reasoning span dropped, then first block
    assert (
        extract_code(
            "<think>```python\nbad()\n```</think>\n```python\ngood()\n```"
        ).strip()
        == "good()"
    )
    assert extract_code("no fenced code") == ""


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


# ------------------ get_run_status edit: exit code is authoritative ------------------ #


@pytest.mark.parametrize(
    "keyword", ["ValueError", "SyntaxError", "out of memory", "TimeoutExpired"]
)
def test_correct_program_with_stderr_keyword_still_passes(keyword):
    # The central get_run_status edit: a correct program (exit 0, matching stdout) that writes
    # an error keyword to stderr must still pass, not be failed on the substring.
    ev = MultilingualLCBEvaluator(timeout_seconds=10.0)
    inst = _instance("python", ["2 3\n"], ["5\n"])
    gen = (
        "```python\nimport sys\n"
        f"sys.stderr.write('{keyword}: not really\\n')\n"
        "a,b=map(int,input().split())\nprint(a+b)\n```"
    )
    assert ev.evaluate_sample(inst, gen).score == 1.0


def test_runtime_error_nonzero_exit_fails():
    solved, meta = LocalSubprocessExecutor().run(
        "raise ValueError('boom')", ["2 3\n"], ["5\n"], "python", 6
    )
    assert solved is False and meta["per_test"] == ["EXECFAIL"]


# ------------------ executor metadata + timeout + early returns ------------------ #


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


# ------------------ pass@k math ------------------ #


@pytest.mark.parametrize("n,c,k", [(0, 0, 1), (5, 0, 0), (5, 6, 1), (5, -1, 1)])
def test_pass_at_k_invalid_raises(n, c, k):
    with pytest.raises(ValueError):
        pass_at_k(n, c, k)


def test_pass_at_k_fractional():
    # n=5, c=2, k=2: 1 - C(3,2)/C(5,2) = 1 - 3/10 = 0.7
    assert pass_at_k(5, 2, 2) == pytest.approx(0.7)


@pytest.mark.parametrize(
    "scores,k,expected", [([1, 0, 1], 1, 2 / 3), ([0, 0, 0], 1, 0.0), ([1, 1], 1, 1.0)]
)
def test_pass_at_k_from_scores(scores, k, expected):
    assert pass_at_k_from_scores(scores, k) == pytest.approx(expected)


def test_pass_at_k_from_scores_empty_raises():
    with pytest.raises(ValueError):
        pass_at_k_from_scores([], 1)


# ------------------ prompt nudges + byte-parity map cross-check ------------------ #

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
def test_mainstream_has_no_nudge(lang):
    # Byte-parity invariant: a mainstream system message equals SYSTEM_MESSAGE_GENERIC alone.
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
    # The hand-copied PLang maps in this test must track the registry, else parity drifts.
    for plang, disp in _PLANG_DISPLAY.items():
        assert LANGUAGES[plang].display == disp
        assert LANGUAGES[plang].md_fence == _PLANG_FENCE.get(plang, plang)


# ------------------ evaluator caching + prepare-once ------------------ #


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
    ev.evaluate_sample(inst, gen)  # same instance+code is a cache hit
    assert ex.runs == 1
    ev.evaluate_sample(inst, "```python\nprint(2)\n```")  # different code reruns
    assert ex.runs == 2
    assert ex.prepared == ["python"]


# ------------------ format_multilingual_prompt (token-id entry point) ------------------ #


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
    assert tok.encode_kw["add_special_tokens"] is False  # avoid a second BOS


# ------------------ multi-test parity + partial pass ------------------ #


def test_parity_multi_test_and_partial():
    inputs = ["2 3\n", "10 20\n", "0 0\n"]
    outputs = [
        "5 \n",
        "30\n",
        "0\n",
    ]  # trailing space on the first to exercise normalization
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
    assert ev.evaluate_sample(inst, partial).score == 0.0  # fails the 0 0 case


# ------------------ structural wiring guard ------------------ #


def test_exact_grading_is_stricter_than_lenient():
    # A python program printing "true" against expected "True": the lenient Multi-LCB
    # comparator aliases True/true (pass), the Agnostics exact comparator does not (fail).
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
    # both agree on an exact-correct answer
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
    from genlm.eval.domains.livecodebench_multilingual.vendored import testing_plang

    assert set(LANGUAGES) <= set(_TOOLCHAIN)
    for lang in LANGUAGES:
        fn, ext = testing_plang.eval_scripts[lang]
        assert callable(fn) and isinstance(ext, str) and ext.startswith(".")
