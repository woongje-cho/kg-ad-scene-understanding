#!/usr/bin/env python3
"""Quick test: 6 queries (1 per category) x 2 trials with new IDs and context-only constraint."""

import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from experiment_v2 import (
    retrieve_kg_context, retrieve_dsg_context,
    get_llm_answer, judge_answer, QUERIES
)
import numpy as np
import time

# 1 per category: Q1(spatial), Q5(identification), Q8(semantic), Q15(hierarchy), Q17(safety), Q24(dilemma)
SAMPLE_IDS = [1, 5, 8, 15, 17, 24]
N_TRIALS = 2

print("=" * 70)
print("v2 Test: 6 queries x 2 modes x 2 trials (context-only constraint)")
print("=" * 70)

results = []
for qid in SAMPLE_IDS:
    q = QUERIES[qid - 1]
    print(f"\n--- Q{qid} [{q['category']}] {q['question'][:55]}...")

    kg_ctx = retrieve_kg_context(qid, q["question"])
    dsg_ctx = retrieve_dsg_context(qid, q["question"])
    print(f"  Context: KG={len(kg_ctx)}ch, 3DSG={len(dsg_ctx)}ch")

    kg_scores, dsg_scores = [], []
    for t in range(N_TRIALS):
        kg_ans = get_llm_answer(kg_ctx, q["question"])
        kg_j = judge_answer(q["question"], q["key_points"], kg_ans)
        kg_scores.append(kg_j.get("score", 0))

        dsg_ans = get_llm_answer(dsg_ctx, q["question"])
        dsg_j = judge_answer(q["question"], q["key_points"], dsg_ans)
        dsg_scores.append(dsg_j.get("score", 0))
        time.sleep(0.2)

    kg_m, dsg_m = np.mean(kg_scores), np.mean(dsg_scores)
    winner = "KG" if kg_m > dsg_m + 0.1 else "3DSG" if dsg_m > kg_m + 0.1 else "TIE"
    print(f"  KG: {kg_scores} avg={kg_m:.1f}  |  3DSG: {dsg_scores} avg={dsg_m:.1f}  -> {winner}")
    print(f"  KG ans: {kg_ans[:150]}...")
    print(f"  3DSG ans: {dsg_ans[:150]}...")
    print(f"  KG judge: {kg_j}")
    print(f"  DSG judge: {dsg_j}")

    results.append({"id": qid, "cat": q["category"], "kg": kg_m, "dsg": dsg_m})

print("\n" + "=" * 70)
print(f"{'Q#':<5} {'Category':<16} {'KG':>5} {'3DSG':>6} {'D':>6} {'Winner':>8}")
print("-" * 50)
for r in results:
    d = r["kg"] - r["dsg"]
    w = "KG" if d > 0.1 else "3DSG" if d < -0.1 else "TIE"
    print(f"Q{r['id']:<4} {r['cat']:<16} {r['kg']:>4.1f} {r['dsg']:>5.1f} {d:>+5.1f} {w:>8}")
print("-" * 50)
kg_avg = np.mean([r["kg"] for r in results])
dsg_avg = np.mean([r["dsg"] for r in results])
print(f"{'AVG':<21} {kg_avg:>4.1f} {dsg_avg:>5.1f} {kg_avg-dsg_avg:>+5.1f}")
