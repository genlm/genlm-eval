#!/usr/bin/env python
"""Offline analysis of the agentic + oracle arms (run on Euler after both finish).

Reports, per model config:
  * quoting rate  -- fraction of generations using >=1 double-quoted identifier
  * agentic retrieval recall vs the official gold tables (selected_tables coverage)
  * fallback rate -- how often the agentic model named nothing parseable

Usage: python analyze_arms.py
"""
import os, glob, json, re

GOLD = os.path.expandvars("$SCRATCH/Spider2/spider2-snow-gold-tables.jsonl")
AGE = os.path.expandvars("$SCRATCH/rollouts/snow_agentic")
ORA = os.path.expandvars("$SCRATCH/rollouts/snow_oracle")
QRE = re.compile(r'"[A-Za-z_][A-Za-z0-9_ ]*"')


def gold_map():
    m = {}
    for line in open(GOLD):
        r = json.loads(line)
        # normalize to lowercase (schema, table) suffix keys
        m[r["instance_id"]] = {tuple(t.lower().split(".")[-2:]) for t in r["gold_tables"]}
    return m


def quoting_rate(root):
    by = {}
    for f in glob.glob(root + "/**/*.jsonl", recursive=True):
        for line in open(f):
            r = json.loads(line)
            k = r["model"].split("/")[-1] + ("-think" if r["thinking"] else "-nothink")
            b = by.setdefault(k, [0, 0])
            for g in r["generations"]:
                b[1] += 1
                if QRE.search(g or ""):
                    b[0] += 1
    return by


def agentic_recall(gm):
    """Per-config: fraction of gold tables covered by selected_tables; fallback rate."""
    by = {}
    for f in glob.glob(AGE + "/**/*.jsonl", recursive=True):
        for line in open(f):
            r = json.loads(line)
            k = r["model"].split("/")[-1] + ("-think" if r["thinking"] else "-nothink")
            gold = gm.get(r["spider2_instance_id"], set())
            if not gold:
                continue
            b = by.setdefault(k, {"cov": 0.0, "full": 0, "n": 0, "fb": 0})
            for i, sel in enumerate(r.get("selected_tables") or []):
                picked = {tuple(s.lower().split(".")[-2:]) for s in (sel or [])}
                hit = len(gold & picked)
                b["cov"] += hit / len(gold)
                b["full"] += int(gold <= picked)
                b["n"] += 1
                if (r.get("retrieval_fallback") or [False] * (i + 1))[i]:
                    b["fb"] += 1
    return by


gm = gold_map()
print("=== QUOTING RATE (>=1 quoted identifier) ===")
for arm, root in [("ORACLE", ORA), ("AGENTIC", AGE)]:
    print(f"-- {arm} --")
    for k, (q, t) in sorted(quoting_rate(root).items()):
        print("  %-22s %5.1f%%   n=%d" % (k, 100 * q / t if t else 0, t))

print("\n=== AGENTIC RETRIEVAL RECALL vs gold tables ===")
print("  %-22s %8s %10s %8s %6s" % ("config", "tbl_recall", "all_tbls%", "fallbk%", "n"))
for k, b in sorted(agentic_recall(gm).items()):
    n = b["n"] or 1
    print("  %-22s %7.1f%% %9.1f%% %7.1f%% %6d"
          % (k, 100 * b["cov"] / n, 100 * b["full"] / n, 100 * b["fb"] / n, b["n"]))
