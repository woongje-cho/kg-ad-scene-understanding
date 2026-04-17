#!/usr/bin/env python3
"""
Extended Ablation Experiment v2: 5-condition comparison
Adds two new conditions to the existing 3-condition ablation:
  1. KG full          — reuse existing results (experiment_results_v2.json)
  2. KG struct-only   — reuse existing results (experiment_results_ablation.json)
  3. 3DSG             — reuse existing results (experiment_results_v2.json)
  4. 3DSG + NL        — NEW: 3DSG context + rdfs:comment descriptions appended
  5. LLM only         — NEW: no scene context, only question + system prompt

Only conditions 4 and 5 need new API calls (30 queries × 5 trials × 2 = 300 calls each).
"""

import json
import os
import time
import numpy as np
from scipy import stats
from openai import OpenAI

# Import from experiment_v2
import experiment_v2 as exp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(SCRIPT_DIR, "experiment_results_ablation_v2.json")
EXISTING_RESULTS_PATH = os.path.join(SCRIPT_DIR, "experiment_results_v2.json")
ABLATION_V1_PATH = os.path.join(SCRIPT_DIR, "experiment_results_ablation.json")

client = OpenAI()


def get_rdfs_comments_for_dsg_entities(query_id: int) -> dict:
    """Fetch rdfs:comment from KG for entities relevant to a query.
    Returns dict mapping entity_id -> comment text."""
    config = exp.KG_ENTITY_MAP.get(query_id, {})
    comments = {}

    # Fetch comments for specific entities
    entity_ids = list(config.get("entities", []))
    entity_ids.extend(config.get("extra_entities", []))

    # Always include ego
    if "ego_car" not in entity_ids:
        entity_ids.insert(0, "ego_car")

    for eid in entity_ids:
        results = exp.sparql_query(
            f"SELECT ?comment WHERE {{ :{eid} rdfs:comment ?comment }} LIMIT 1"
        )
        if results and results[0].get("comment"):
            comments[eid] = results[0]["comment"]

    # If query uses nearby_ego, zone_objects, etc., fetch comments for those too
    if config.get("nearby_ego"):
        nearby = exp.sparql_query(
            """SELECT ?obj ?comment WHERE {
                ?obj rdfs:comment ?comment .
            } LIMIT 25"""
        )
        for n in nearby:
            comments[n["obj"]] = n["comment"]

    if config.get("zones"):
        for zone_id in config["zones"]:
            if config.get("zone_objects"):
                objs = exp.sparql_query(
                    f"""SELECT ?obj ?comment WHERE {{
                        ?obj :inZone :{zone_id} ; rdfs:comment ?comment .
                    }}"""
                )
                for o in objs:
                    comments[o["obj"]] = o["comment"]

    if config.get("all_vehicles"):
        vehicles = exp.sparql_query(
            """SELECT ?v ?comment WHERE {
                ?v rdf:type/rdfs:subClassOf* :Vehicle ; rdfs:comment ?comment .
            }"""
        )
        for v in vehicles:
            comments[v["v"]] = v["comment"]

    if config.get("all_pedestrians"):
        peds = exp.sparql_query(
            """SELECT ?p ?comment WHERE {
                ?p rdf:type/rdfs:subClassOf* :Pedestrian ; rdfs:comment ?comment .
            }"""
        )
        for p in peds:
            comments[p["p"]] = p["comment"]

    if config.get("scene_overview"):
        overview = exp.sparql_query(
            """SELECT ?obj ?comment WHERE {
                ?obj rdfs:comment ?comment .
            } LIMIT 15"""
        )
        for o in overview:
            comments[o["obj"]] = o["comment"]

    return comments


def augment_dsg_with_nl(dsg_context: str, comments: dict) -> str:
    """Append rdfs:comment descriptions to 3DSG context for each matched entity."""
    lines = dsg_context.split("\n")
    augmented = []
    for line in lines:
        augmented.append(line)
        # Check if this line contains an entity ID we have a comment for
        for eid, comment in comments.items():
            if eid in line and "description:" not in line.lower():
                # Only add if it's a main entity line (contains class= or type=)
                if "class=" in line or "type=" in line or f"[{eid}]" in line or eid + " " in line:
                    augmented.append(f"    description: {comment[:300]}")
                    break
    return "\n".join(augmented)


def run_extended_ablation():
    print("=" * 80)
    print("EXTENDED ABLATION v2: 5-condition comparison")
    print(f"Model: {exp.ANSWER_MODEL} (answer) / {exp.JUDGE_MODEL} (judge)")
    print(f"Trials: {exp.N_TRIALS} per query")
    print("=" * 80)

    # Load existing results
    with open(EXISTING_RESULTS_PATH, "r", encoding="utf-8") as f:
        existing = json.load(f)
    existing_queries = {q["id"]: q for q in existing["queries"]}

    with open(ABLATION_V1_PATH, "r", encoding="utf-8") as f:
        ablation_v1 = json.load(f)
    ablation_v1_queries = {q["id"]: q for q in ablation_v1["queries"]}

    all_results = []
    total_calls = 0

    for q in exp.QUERIES:
        qid = q["id"]
        question = q["question"]
        key_points = q["key_points"]
        category = q["category"]

        print(f"\n--- Q{qid} [{category}] {question[:60]}...")

        # Retrieve 3DSG context and rdfs:comment lookup
        dsg_ctx = exp.retrieve_dsg_context(qid, question)
        comments = get_rdfs_comments_for_dsg_entities(qid)
        dsg_nl_ctx = augment_dsg_with_nl(dsg_ctx, comments)

        print(f"  3DSG: {len(dsg_ctx)} chars | 3DSG+NL: {len(dsg_nl_ctx)} chars (+{len(dsg_nl_ctx)-len(dsg_ctx)})")
        print(f"  Comments found: {len(comments)} entities")

        q_result = {
            "id": qid,
            "category": category,
            "question": question,
            # New condition 4: 3DSG + NL
            "dsg_nl_scores": [],
            "dsg_nl_answers": [],
            "dsg_nl_judgments": [],
            # New condition 5: LLM only
            "llm_only_scores": [],
            "llm_only_answers": [],
            "llm_only_judgments": [],
            # Context info
            "dsg_ctx_len": len(dsg_ctx),
            "dsg_nl_ctx_len": len(dsg_nl_ctx),
            "n_comments": len(comments),
        }

        # === Condition 4: 3DSG + NL ===
        for trial in range(exp.N_TRIALS):
            answer = exp.get_llm_answer(dsg_nl_ctx, question)
            judgment = exp.judge_answer(question, key_points, answer)
            q_result["dsg_nl_answers"].append(answer)
            q_result["dsg_nl_judgments"].append(judgment)
            q_result["dsg_nl_scores"].append(judgment.get("score", 0))
            total_calls += 2
            time.sleep(0.2)

        # === Condition 5: LLM only ===
        no_context = (
            "No scene context is provided. "
            "You must answer based solely on general knowledge about autonomous driving."
        )
        for trial in range(exp.N_TRIALS):
            answer = exp.get_llm_answer(no_context, question)
            judgment = exp.judge_answer(question, key_points, answer)
            q_result["llm_only_answers"].append(answer)
            q_result["llm_only_judgments"].append(judgment)
            q_result["llm_only_scores"].append(judgment.get("score", 0))
            total_calls += 2
            time.sleep(0.2)

        # Get existing scores
        ex = existing_queries.get(qid, {})
        ab = ablation_v1_queries.get(qid, {})

        kg_full_mean = ex.get("kg_mean", 0)
        dsg_mean = ex.get("dsg_mean", 0)
        kg_nodesc_mean = ab.get("kg_nodesc_mean", 0)
        dsg_nl_mean = float(np.mean(q_result["dsg_nl_scores"]))
        llm_only_mean = float(np.mean(q_result["llm_only_scores"]))

        q_result["kg_full_mean"] = kg_full_mean
        q_result["kg_nodesc_mean"] = kg_nodesc_mean
        q_result["dsg_mean"] = dsg_mean
        q_result["dsg_nl_mean"] = round(dsg_nl_mean, 2)
        q_result["llm_only_mean"] = round(llm_only_mean, 2)

        print(f"  LLM-only: {llm_only_mean:.1f} | KG-struct: {kg_nodesc_mean:.1f} | "
              f"3DSG: {dsg_mean:.1f} | 3DSG+NL: {dsg_nl_mean:.1f} | KG-full: {kg_full_mean:.1f}")

        all_results.append(q_result)

    # ============================================================
    # Analysis
    # ============================================================
    print("\n" + "=" * 80)
    print("5-CONDITION ABLATION RESULTS")
    print("=" * 80)

    categories = ["spatial", "identification", "semantic", "hierarchy", "safety", "dilemma"]

    print(f"\n{'Category':<16} {'LLM-only':>9} {'KG-struct':>10} {'3DSG':>7} {'3DSG+NL':>8} {'KG-full':>8}")
    print("-" * 65)

    cat_stats = {}
    for cat in categories:
        cat_results = [r for r in all_results if r["category"] == cat]
        vals = {
            "llm_only": np.mean([r["llm_only_mean"] for r in cat_results]),
            "kg_nodesc": np.mean([r["kg_nodesc_mean"] for r in cat_results]),
            "dsg": np.mean([r["dsg_mean"] for r in cat_results]),
            "dsg_nl": np.mean([r["dsg_nl_mean"] for r in cat_results]),
            "kg_full": np.mean([r["kg_full_mean"] for r in cat_results]),
        }
        cat_stats[cat] = {k: round(float(v), 3) for k, v in vals.items()}
        print(f"{cat:<16} {vals['llm_only']:>8.2f} {vals['kg_nodesc']:>9.2f} "
              f"{vals['dsg']:>6.2f} {vals['dsg_nl']:>7.2f} {vals['kg_full']:>7.2f}")

    # Overall
    overall = {
        "llm_only": float(np.mean([r["llm_only_mean"] for r in all_results])),
        "kg_nodesc": float(np.mean([r["kg_nodesc_mean"] for r in all_results])),
        "dsg": float(np.mean([r["dsg_mean"] for r in all_results])),
        "dsg_nl": float(np.mean([r["dsg_nl_mean"] for r in all_results])),
        "kg_full": float(np.mean([r["kg_full_mean"] for r in all_results])),
    }
    print("-" * 65)
    print(f"{'OVERALL':<16} {overall['llm_only']:>8.2f} {overall['kg_nodesc']:>9.2f} "
          f"{overall['dsg']:>6.2f} {overall['dsg_nl']:>7.2f} {overall['kg_full']:>7.2f}")

    # Statistical tests
    print("\n--- Pairwise Statistical Tests (Wilcoxon signed-rank) ---")
    pairs = [
        ("KG-full vs 3DSG+NL", [r["kg_full_mean"] for r in all_results], [r["dsg_nl_mean"] for r in all_results]),
        ("3DSG+NL vs 3DSG", [r["dsg_nl_mean"] for r in all_results], [r["dsg_mean"] for r in all_results]),
        ("3DSG vs KG-struct", [r["dsg_mean"] for r in all_results], [r["kg_nodesc_mean"] for r in all_results]),
        ("KG-struct vs LLM-only", [r["kg_nodesc_mean"] for r in all_results], [r["llm_only_mean"] for r in all_results]),
        ("KG-full vs LLM-only", [r["kg_full_mean"] for r in all_results], [r["llm_only_mean"] for r in all_results]),
    ]

    stat_results = {}
    for name, a, b in pairs:
        try:
            stat, p = stats.wilcoxon(a, b, alternative="greater")
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            diff = np.array(a) - np.array(b)
            d = float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else 0
            print(f"  {name:<25} W={stat:.1f}, p={p:.6f} {sig}, d={d:.3f}")
            stat_results[name] = {"W": float(stat), "p": float(p), "sig": sig, "d": round(d, 3)}
        except Exception as e:
            print(f"  {name}: error — {e}")
            stat_results[name] = {"error": str(e)}

    # Per-query table
    print(f"\n{'Q#':<4} {'Cat':<14} {'LLM':>5} {'KG-s':>6} {'3DSG':>6} {'3D+NL':>6} {'KG-f':>6}")
    print("-" * 50)
    for r in all_results:
        print(f"Q{r['id']:<3} {r['category']:<14} {r['llm_only_mean']:>4.1f} {r['kg_nodesc_mean']:>5.1f} "
              f"{r['dsg_mean']:>5.1f} {r['dsg_nl_mean']:>5.1f} {r['kg_full_mean']:>5.1f}")

    # Save results
    output = {
        "config": {
            "experiment": "ablation_v2_5conditions",
            "answer_model": exp.ANSWER_MODEL,
            "judge_model": exp.JUDGE_MODEL,
            "n_trials": exp.N_TRIALS,
            "answer_temp": exp.ANSWER_TEMP,
            "conditions": [
                "LLM-only (no context)",
                "KG structure-only (no rdfs:comment)",
                "3DSG (baseline)",
                "3DSG + NL annotation (3DSG + rdfs:comment)",
                "KG full (structure + rdfs:comment)",
            ],
        },
        "summary": {
            "overall": {k: round(v, 3) for k, v in overall.items()},
            "categories": cat_stats,
            "statistical_tests": stat_results,
        },
        "queries": [
            {
                "id": r["id"],
                "category": r["category"],
                "question": r["question"],
                "kg_full_mean": r["kg_full_mean"],
                "kg_nodesc_mean": r["kg_nodesc_mean"],
                "dsg_mean": r["dsg_mean"],
                "dsg_nl_mean": r["dsg_nl_mean"],
                "dsg_nl_scores": r["dsg_nl_scores"],
                "llm_only_mean": r["llm_only_mean"],
                "llm_only_scores": r["llm_only_scores"],
                "dsg_nl_answers": r["dsg_nl_answers"],
                "llm_only_answers": r["llm_only_answers"],
                "dsg_nl_judgments": r["dsg_nl_judgments"],
                "llm_only_judgments": r["llm_only_judgments"],
                "n_comments": r["n_comments"],
            }
            for r in all_results
        ],
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nTotal NEW API calls: {total_calls}")
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run_extended_ablation()
