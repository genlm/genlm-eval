#!/usr/bin/env python
"""Merge per-shard rollout JSONL files into one file per (model, temperature).

The generation worker writes ``<slug>__t<temp>__shard<i>-of<N>.jsonl`` per GPU
task; this concatenates the shards back into ``<slug>__t<temp>.jsonl`` (sorted by
instance_id) so each (model, temperature) is a single file again.

    python scripts/merge_rollout_shards.py --out-dir rollouts/spider2-snow
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

SHARD_RE = re.compile(r"^(?P<base>.+)__shard\d+-of\d+\.jsonl$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="Rollout root (contains <slug>/ dirs).")
    ap.add_argument("--keep-shards", action="store_true", help="Do not delete shard files after merging.")
    args = ap.parse_args()

    root = Path(args.out_dir)
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        groups: dict[str, list[Path]] = defaultdict(list)
        for f in model_dir.glob("*__shard*-of*.jsonl"):
            m = SHARD_RE.match(f.name)
            if m:
                groups[m.group("base")].append(f)

        for base, shards in sorted(groups.items()):
            rows = []
            for shard in shards:
                with open(shard, encoding="utf-8") as fh:
                    rows.extend(json.loads(line) for line in fh if line.strip())
            rows.sort(key=lambda r: r["instance_id"])
            merged = model_dir / f"{base}.jsonl"
            with open(merged, "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            print(f"{merged}: {len(rows)} instances from {len(shards)} shards")
            if not args.keep_shards:
                for shard in shards:
                    shard.unlink()


if __name__ == "__main__":
    main()
