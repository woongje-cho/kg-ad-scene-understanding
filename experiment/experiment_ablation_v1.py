#!/usr/bin/env python3
"""
Ablation Experiment: KG without rdfs:comment vs KG full vs 3DSG
Tests whether semantic descriptions (rdfs:comment) are the key factor
in KG's advantage over 3DSG.

Three conditions:
  1. KG full     — reuse existing results from experiment_results_v2.json
  2. KG no-desc  — KG context with rdfs:comment stripped (NEW experiment)
  3. 3DSG        — reuse existing results from experiment_results_v2.json

Only condition 2 needs new API calls (30 queries × 5 trials = 150 calls).
"""

import json
import os
import re
import time
import numpy as np
from scipy import stats
from openai import OpenAI

# Import from experiment_v2
import experiment_v2 as exp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(SCRIPT_DIR, "experiment_results_ablation.json")
EXISTING_RESULTS_PATH = os.path.join(SCRIPT_DIR, "experiment_results_v2.json")

client = OpenAI()


def strip_descriptions(context: str) -> str:
    """Remove all rdfs:comment / description lines from KG context.

    Strips lines starting with:
      - '  description: ...'
      - '    desc: ...'
    Also removes multi-line descriptions that continue on next lines
    (indented continuation).
    """
    lines = context.split("\n")
    filtered = []
    skip_continuation = False

    for line in lines:
        # Check if this is a description/desc line
        stripped = line.lstrip()
        if stripped.startswith("description:") or stripped.startswith("desc:"):
            skip_continuation = True
            continue

        # Check if this is a continuation of a skipped description
        # (indented more than the previous non-desc line)
        if skip_continuation:
            # If line is empty or starts with a known pattern, stop skipping
            if (not stripped or
                stripped.startswith("[") or
                stripped.startswith("edge:") or
                re.match(r'^[a-zA-Z_]', stripped)):
                skip_continuation = False
                filtered.append(line)
            # else: still continuation, skip
        else:
            filtered.append(line)

    return "\n".join(filtered)


def run_ablation():
    print("=" * 80)
    print("ABLATION: KG without rdfs:comment")
    print(f"Model: {exp.ANSWER_MODEL} (answer) / {exp.JUDGE_MODEL} (judge)")
    print(f"Trials: {exp.N_TRIALS} per query")
    print("=" * 80)

    # Load existing results for comparison
    with open(EXISTING_RESULTS_PATH, "r", encoding="utf-8") as f:
        existing = json.load(f)

    existing_queries = {q["id"]: q for q in existing["queries"]}

    all_results = []
    total_calls = 0

    for q in exp.QUERIES:
        qid = q["id"]
        question = q["question"]
        key_points = q["key_points"]
        category = q["category"]

        print(f"\n--- Q{qid} [{category}] {question[:60]}...")

        # Get KG context and strip descriptions
        kg_ctx_full = exp.retrieve_kg_context(qid, question)
        kg_ctx_no_desc = strip_descriptions(kg_ctx_full)

        # Show context length reduction
        full_len = len(kg_ctx_full)
        nodesc_len = len(kg_ctx_no_desc)
        reduction = (1 - nodesc_len / full_len) * 100 if full_len > 0 else 0
        print(f"  Context: {full_len} → {nodesc_len} chars ({reduction:.0f}% reduction)")

        q_result = {
            "id": qid,
            "category": category,
            "question": question,
            "kg_nodesc_scores": [],
            "kg_nodesc_answers": [],
            "kg_nodesc_judgments": [],
            "kg_full_context": kg_ctx_full[:500],
            "kg_nodesc_context": kg_ctx_no_desc[:500],
            "context_reduction_pct": round(reduction, 1),
        }

        # Run trials for KG no-description
        for trial in range(exp.N_TRIALS):
            answer = exp.get_llm_answer(kg_ctx_no_desc, question)
            judgment = exp.judge_answer(question, key_points, answer)
            q_result["kg_nodesc_answers"].append(answer)
            q_result["kg_nodesc_judgments"].append(judgment)
            q_result["kg_nodesc_scores"].append(judgment.get("score", 0))
            total_calls += 2  # 1 answer + 1 judgment
            time.sleep(0.2)

        # Get existing scores for comparison
        ex = existing_queries.get(qid, {})
        kg_full_mean = ex.get("kg_mean", 0)
        dsg_mean = ex.get("dsg_mean", 0)
        nodesc_mean = np.mean(q_result["kg_nodesc_scores"])
        nodesc_std = np.std(q_result["kg_nodesc_scores"])

        print(f"  KG full: {kg_full_mean:.1f}  |  KG no-desc: {nodesc_mean:.1f} ± {nodesc_std:.1f}  |  3DSG: {dsg_mean:.1f}")

        q_result["kg_full_mean_existing"] = kg_full_mean
        q_result["dsg_mean_existing"] = dsg_mean
        q_result["kg_nodesc_mean"] = round(float(nodesc_mean), 2)

        all_results.append(q_result)

    # ============================================================
    # Analysis
    # ============================================================
    print("\n" + "=" * 80)
    print("ABLATION RESULTS: KG full vs KG no-desc vs 3DSG")
    print("=" * 80)

    categories = ["spatial", "identification", "semantic", "hierarchy", "safety", "dilemma"]

    print(f"\n{'Category':<20} {'KG full':>8} {'KG no-desc':>11} {'3DSG':>8} {'Δ(full-nodesc)':>15}")
    print("-" * 70)

    cat_stats = {}
    for cat in categories:
        cat_results = [r for r in all_results if r["category"] == cat]
        kg_full = [r["kg_full_mean_existing"] for r in cat_results]
        kg_nodesc = [r["kg_nodesc_mean"] for r in cat_results]
        dsg = [r["dsg_mean_existing"] for r in cat_results]

        kg_full_avg = np.mean(kg_full)
        kg_nodesc_avg = np.mean(kg_nodesc)
        dsg_avg = np.mean(dsg)
        delta = kg_full_avg - kg_nodesc_avg

        cat_stats[cat] = {
            "kg_full": round(float(kg_full_avg), 3),
            "kg_nodesc": round(float(kg_nodesc_avg), 3),
            "dsg": round(float(dsg_avg), 3),
            "delta_full_nodesc": round(float(delta), 3),
        }

        print(f"{cat:<20} {kg_full_avg:>7.2f} {kg_nodesc_avg:>10.2f} {dsg_avg:>7.2f} {delta:>+14.2f}")

    # Overall
    all_kg_full = [r["kg_full_mean_existing"] for r in all_results]
    all_kg_nodesc = [r["kg_nodesc_mean"] for r in all_results]
    all_dsg = [r["dsg_mean_existing"] for r in all_results]

    overall_full = np.mean(all_kg_full)
    overall_nodesc = np.mean(all_kg_nodesc)
    overall_dsg = np.mean(all_dsg)

    print("-" * 70)
    print(f"{'OVERALL':<20} {overall_full:>7.2f} {overall_nodesc:>10.2f} {overall_dsg:>7.2f} {overall_full-overall_nodesc:>+14.2f}")

    # Statistical tests
    print("\n--- Statistical Tests ---")

    # Test 1: KG full vs KG no-desc (paired)
    try:
        stat, p = stats.wilcoxon(all_kg_full, all_kg_nodesc, alternative="greater")
        print(f"KG full vs KG no-desc: W={stat:.1f}, p={p:.6f}")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  → {sig}")
    except Exception as e:
        print(f"  Wilcoxon error: {e}")

    # Test 2: KG no-desc vs 3DSG (paired)
    try:
        stat, p = stats.wilcoxon(all_kg_nodesc, all_dsg, alternative="greater")
        print(f"KG no-desc vs 3DSG:   W={stat:.1f}, p={p:.6f}")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  → {sig}")
    except Exception as e:
        print(f"  Wilcoxon error: {e}")

    # Effect sizes
    print("\n--- Effect Sizes (Cohen's d) ---")
    def cohens_d(a, b):
        diff = np.array(a) - np.array(b)
        return np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0

    d_full_nodesc = cohens_d(all_kg_full, all_kg_nodesc)
    d_nodesc_dsg = cohens_d(all_kg_nodesc, all_dsg)
    d_full_dsg = cohens_d(all_kg_full, all_dsg)

    print(f"  KG full vs KG no-desc: d={d_full_nodesc:.3f}")
    print(f"  KG no-desc vs 3DSG:    d={d_nodesc_dsg:.3f}")
    print(f"  KG full vs 3DSG:       d={d_full_dsg:.3f}")

    # Context reduction stats
    avg_reduction = np.mean([r["context_reduction_pct"] for r in all_results])
    print(f"\n--- Context Length ---")
    print(f"  Average reduction when removing descriptions: {avg_reduction:.1f}%")

    # Per-query table
    print(f"\n{'Q#':<4} {'Category':<16} {'KG full':>8} {'KG no-desc':>11} {'3DSG':>6} {'Δ(f-nd)':>8}")
    print("-" * 60)
    for r in all_results:
        delta = r["kg_full_mean_existing"] - r["kg_nodesc_mean"]
        print(f"Q{r['id']:<3} {r['category']:<16} {r['kg_full_mean_existing']:>7.1f} {r['kg_nodesc_mean']:>10.1f} {r['dsg_mean_existing']:>5.1f} {delta:>+7.1f}")

    # Save results
    output = {
        "config": {
            "experiment": "ablation_rdfs_comment",
            "answer_model": exp.ANSWER_MODEL,
            "judge_model": exp.JUDGE_MODEL,
            "n_trials": exp.N_TRIALS,
            "answer_temp": exp.ANSWER_TEMP,
            "description": "KG context with rdfs:comment descriptions stripped. "
                          "Tests whether semantic descriptions are the key factor.",
        },
        "summary": {
            "overall_kg_full": round(float(overall_full), 3),
            "overall_kg_nodesc": round(float(overall_nodesc), 3),
            "overall_dsg": round(float(overall_dsg), 3),
            "delta_full_nodesc": round(float(overall_full - overall_nodesc), 3),
            "delta_nodesc_dsg": round(float(overall_nodesc - overall_dsg), 3),
            "cohens_d_full_nodesc": round(float(d_full_nodesc), 3),
            "cohens_d_nodesc_dsg": round(float(d_nodesc_dsg), 3),
            "avg_context_reduction_pct": round(float(avg_reduction), 1),
            "categories": cat_stats,
        },
        "queries": [
            {
                "id": r["id"],
                "category": r["category"],
                "question": r["question"],
                "kg_nodesc_scores": r["kg_nodesc_scores"],
                "kg_nodesc_mean": r["kg_nodesc_mean"],
                "kg_full_mean_existing": r["kg_full_mean_existing"],
                "dsg_mean_existing": r["dsg_mean_existing"],
                "kg_nodesc_answers": r["kg_nodesc_answers"],
                "kg_nodesc_judgments": r["kg_nodesc_judgments"],
                "kg_full_context_preview": r["kg_full_context"],
                "kg_nodesc_context_preview": r["kg_nodesc_context"],
                "context_reduction_pct": r["context_reduction_pct"],
            }
            for r in all_results
        ],
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nTotal NEW LLM calls: {total_calls}")
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run_ablation()
