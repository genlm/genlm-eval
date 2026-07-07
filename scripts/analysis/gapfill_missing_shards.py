import os
base = os.path.expandvars("$SCRATCH/rollouts/snow_oracle_n100")
SLUGS = ["qwen3-4b-think","qwen3-4b-nothink","qwen3-8b-think","qwen3-8b-nothink",
         "qwen3-1.7b-think","qwen3-1.7b-nothink"]
NS = 128
missing = []
for mi, slug in enumerate(SLUGS):
    for shard in range(NS):
        fn = f"{base}/{slug}/{slug}__t0.6__shard{shard:03d}-of{NS:03d}__oracle.jsonl"
        if not os.path.exists(fn):
            missing.append(mi*NS + shard)
present = 6*NS - len(missing)
print(f"present={present}/768  missing={len(missing)}")
# 8B range (256-511) needs TP=2 / 2 GPUs; the rest 1 GPU
eightb = sorted(t for t in missing if 256 <= t <= 511)
other  = sorted(t for t in missing if not (256 <= t <= 511))
def fmt(xs): return ",".join(map(str, xs))
print("OTHER_1GPU:", fmt(other) if other else "(none)")
print("EIGHTB_TP2:", fmt(eightb) if eightb else "(none)")
