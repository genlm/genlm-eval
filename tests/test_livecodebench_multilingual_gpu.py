"""Opt-in GPU end-to-end test for the multilingual LCB domain.

Validates the full pipeline with a real code model on GPU: build our prompt, generate, run
our extractor and grader, and report pass@1 via run_evaluation. Heavy (downloads a ~1GB model
and needs a GPU), so it is skipped unless CUDA is available and GENLM_GPU_TEST=1 is set:

    GENLM_GPU_TEST=1 python -m pytest tests/test_livecodebench_multilingual_gpu.py -v

Model is configurable via GENLM_GPU_TEST_MODEL (default Qwen/Qwen2.5-Coder-0.5B-Instruct).
"""

import os

import pytest

_GPU_OPT_IN = os.environ.get("GENLM_GPU_TEST") == "1"
_MODEL = os.environ.get("GENLM_GPU_TEST_MODEL", "Qwen/Qwen2.5-Coder-0.5B-Instruct")
FIXTURE = "tests/fixtures/lcb_sample.jsonl"


def _cuda_available():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not (_GPU_OPT_IN and _cuda_available()),
    reason="set GENLM_GPU_TEST=1 and run on a CUDA node",
)


@pytest.mark.parametrize("language", ["python", "c++"])
def test_end_to_end_generation_and_grading(language):
    import asyncio

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from genlm.eval import ModelOutput, ModelResponse, run_evaluation
    from genlm.eval.domains.livecodebench_multilingual import (
        MultilingualLCBDataset,
        MultilingualLCBEvaluator,
        multilingual_chat_messages,
    )

    tok = AutoTokenizer.from_pretrained(_MODEL)
    model = (
        AutoModelForCausalLM.from_pretrained(_MODEL, dtype=torch.float16)
        .to("cuda")
        .eval()
    )

    async def adapter(instance, output_dir, replicate):
        text = tok.apply_chat_template(
            multilingual_chat_messages(instance),
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        gen = tok.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
        return ModelOutput(responses=[ModelResponse(response=gen, weight=1.0)])

    ds = MultilingualLCBDataset.from_jsonl(FIXTURE, language)
    res = asyncio.run(
        run_evaluation(
            dataset=ds,
            model=adapter,
            evaluator=MultilingualLCBEvaluator(timeout_seconds=10.0),
            max_instances=2,
            n_replicates=1,
        )
    )
    # Pipeline ran end-to-end and produced a well-formed result: strict 0/1 scores, every
    # response tagged with the right language, and a real executor status (not None).
    assert res["n_instances"] == 2
    for inst_results in res["all_instance_results"]:
        result = inst_results[0]
        assert result["weighted_accuracy"] in (0.0, 1.0)
        for resp in result["results"]:
            assert resp["score"] in (0.0, 1.0)
            assert resp["metadata"]["language"] == language
            assert resp["metadata"]["status"] is not None
