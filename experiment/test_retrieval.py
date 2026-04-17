#!/usr/bin/env python3
"""Test KG vs DSG retrieval for problematic queries."""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from experiment_v2 import retrieve_kg_context, retrieve_dsg_context

for qid in [3, 4]:
    print(f"\n{'='*60}")
    print(f"Q{qid} KG Context:")
    print(f"{'='*60}")
    kg = retrieve_kg_context(qid, "test")
    print(kg)
    print(f"\n{'='*60}")
    print(f"Q{qid} DSG Context:")
    print(f"{'='*60}")
    dsg = retrieve_dsg_context(qid, "test")
    print(dsg)
