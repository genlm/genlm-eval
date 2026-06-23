"""Registry of the target languages for multilingual LiveCodeBench.

The 12 mainstream languages come from Multi-LCB; their ``display``, ``md_fence`` and
``comment`` values match ``lcb_runner.utils.PLang`` exactly so the generated prompts are
byte-identical to Multi-LCB's (prompt parity). The 5 low-resource languages come from the
Agnostics project (github.com/nuprl/agnostics-framework); their ``prompt_nudge`` text is
paraphrased here (Agnostics ships no license), not copied verbatim.

``key`` is the canonical name the vendored ``testing_plang.eval_scripts`` dispatch table
expects; all 17 are wired. Grading a language whose toolchain is not installed raises a clear
error from the executor's ``prepare`` (see ``is_toolchain_available``).
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Language:
    key: str  # canonical eval_scripts key, e.g. "c++"
    display: str  # name shown in the system message, e.g. "C++"
    md_fence: str  # markdown code-fence tag, e.g. "cpp"
    comment: str  # single-line comment token, e.g. "//"
    tier: int  # 1 mainstream, 2 low-resource, 3 high-risk
    source: str  # "multilcb" | "agnostics"
    prompt_nudge: str = ""  # appended language guidance (low-resource langs only)


# 12 Multi-LCB languages. display/md_fence/comment mirror PLang verbatim
# (note: php's display is lowercase "php" upstream; we keep it for parity).
_MULTILCB = [
    Language("python", "Python", "python", "#", 1, "multilcb"),
    Language("c++", "C++", "cpp", "//", 1, "multilcb"),
    Language("java", "Java", "java", "//", 1, "multilcb"),
    Language("c#", "C#", "csharp", "//", 1, "multilcb"),
    Language("go", "Go", "go", "//", 1, "multilcb"),
    Language("javascript", "JavaScript", "javascript", "//", 1, "multilcb"),
    Language("typescript", "TypeScript", "typescript", "//", 1, "multilcb"),
    Language("rust", "Rust", "rust", "//", 1, "multilcb"),
    Language("ruby", "Ruby", "ruby", "#", 1, "multilcb"),
    Language("php", "php", "php", "//", 1, "multilcb"),
    Language("kotlin", "Kotlin", "kotlin", "//", 1, "multilcb"),
    Language("scala", "Scala", "scala", "//", 1, "multilcb"),
]

# 5 Agnostics low-resource languages. Nudges paraphrased from the pl-configs prompt fields.
_AGNOSTICS = [
    Language(
        "lua",
        "Lua",
        "lua",
        "--",
        2,
        "agnostics",
        prompt_nudge="Target Lua 5.1 / LuaJIT.",
    ),
    Language(
        "julia",
        "Julia",
        "julia",
        "#",
        2,
        "agnostics",
        prompt_nudge="Target Julia 1.11.",
    ),
    Language(
        "r",
        "R",
        "r",
        "#",
        2,
        "agnostics",
        prompt_nudge=(
            'Target R 4. Read stdin with readLines(con = file("stdin")) (the optional n '
            "argument limits how many lines are read) and write output with cat; do not use "
            "print."
        ),
    ),
    Language(
        "ocaml",
        "OCaml",
        "ocaml",
        "(*",
        3,
        "agnostics",
        prompt_nudge=(
            "Target OCaml 5 using the standard library for I/O (Scanf/Printf, read_line). "
            "Remember the dotted float operators (+. -. *. /.), explicit int/float casts, "
            "and that lists favour pattern matching or folds over indexing."
        ),
    ),
    Language(
        "fortran",
        "Fortran",
        "fortran",
        "!",
        2,
        "agnostics",
        prompt_nudge=(
            "Target Fortran 90. Begin each scope with implicit none; arrays are 1-based; "
            "read a size before allocating and reading an array; use real literals (e.g. "
            "2.0d0) to avoid integer division; read inputs, compute, and write output only."
        ),
    ),
]

LANGUAGES: Dict[str, Language] = {lang.key: lang for lang in (_MULTILCB + _AGNOSTICS)}

# Convenience aliases accepted by resolve_language (canonical keys also resolve to themselves).
_ALIASES = {
    "cpp": "c++",
    "cplusplus": "c++",
    "csharp": "c#",
    "cs": "c#",
    "js": "javascript",
    "ts": "typescript",
    "golang": "go",
}


def resolve_language(name: str) -> Language:
    """Resolve a user-supplied language name (case-insensitive, with aliases) to a Language.

    Raises ValueError for an unknown language.
    """
    key = name.strip().lower()
    key = _ALIASES.get(key, key)
    if key not in LANGUAGES:
        raise ValueError(
            f"unknown language {name!r}; known: {sorted(LANGUAGES)} "
            f"(aliases: {sorted(_ALIASES)})"
        )
    return LANGUAGES[key]
