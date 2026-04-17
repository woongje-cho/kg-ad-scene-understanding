#!/usr/bin/env python3
"""
Semantic Search Query Tool

Usage:
    # Using query file
    python tools/semantic_search.py queries/semantic/find_robot.json

    # Direct query
    python tools/semantic_search.py --query "find a comfortable place to sit" --top-k 3

    # List all example queries
    python tools/semantic_search.py --list
"""

import json
import requests
import sys
from pathlib import Path
from typing import Optional
import argparse


class SemanticSearchTool:
    """Tool for running semantic search queries."""

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.semantic_search_endpoint = f"{api_url}/semantic_search"

    def search(self, query: str, top_k: int = 5) -> dict:
        """Execute a semantic search query."""
        try:
            response = requests.post(
                self.semantic_search_endpoint,
                params={"query": query, "top_k": top_k}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": str(e)}

    def search_from_file(self, query_file: Path) -> dict:
        """Execute query from a JSON file."""
        with open(query_file, 'r') as f:
            query_data = json.load(f)

        query = query_data.get("query")
        top_k = query_data.get("top_k", 5)

        print(f"📄 Query file: {query_file}")
        print(f"🔍 Query: {query}")
        print(f"📊 Top K: {top_k}")
        if "description" in query_data:
            print(f"💬 Description: {query_data['description']}")
        print()

        return self.search(query, top_k)

    def display_results(self, results: dict):
        """Display search results in a readable format."""
        if results.get("status") == "error":
            print(f"❌ Error: {results.get('message')}")
            return

        print("=" * 80)
        print(f"🔍 Semantic Search Results")
        print("=" * 80)
        print(f"Query: {results.get('query')}")
        print(f"Found: {results.get('count')} results")
        print()

        for i, result in enumerate(results.get("results", []), 1):
            print(f"{i}. {result['id']}")
            print(f"   Types: {', '.join(result['types'])}")
            print(f"   Description: {result['description']}")
            print(f"   Similarity: {result['score']:.4f}")
            print()

    def list_queries(self):
        """List all available example queries."""
        queries_dir = Path("queries/semantic")
        if not queries_dir.exists():
            print("No semantic query examples found.")
            return

        print("=" * 80)
        print("📚 Available Semantic Query Examples")
        print("=" * 80)
        print()

        for query_file in sorted(queries_dir.glob("*.json")):
            with open(query_file, 'r') as f:
                query_data = json.load(f)

            print(f"📄 {query_file.name}")
            print(f"   Query: {query_data.get('query')}")
            if "description" in query_data:
                print(f"   Description: {query_data.get('description')}")
            print(f"   Top K: {query_data.get('top_k', 5)}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Semantic Search Query Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "query_file",
        nargs="?",
        type=Path,
        help="Path to query JSON file"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Direct query string"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available example queries"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="API server URL (default: http://localhost:8000)"
    )

    args = parser.parse_args()

    tool = SemanticSearchTool(api_url=args.api_url)

    # List queries
    if args.list:
        tool.list_queries()
        return

    # Execute query
    if args.query:
        # Direct query
        print(f"🔍 Query: {args.query}")
        print(f"📊 Top K: {args.top_k}")
        print()
        results = tool.search(args.query, args.top_k)
    elif args.query_file:
        # Query from file
        if not args.query_file.exists():
            print(f"❌ Error: Query file not found: {args.query_file}")
            sys.exit(1)
        results = tool.search_from_file(args.query_file)
    else:
        parser.print_help()
        sys.exit(1)

    tool.display_results(results)


if __name__ == "__main__":
    main()
