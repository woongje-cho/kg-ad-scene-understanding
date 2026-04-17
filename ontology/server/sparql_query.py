#!/usr/bin/env python3
"""
SPARQL Query Tool
Execute SPARQL queries on the running ontology server
Reads query selection from queries/config.yaml
"""

import sys
import requests
import yaml
from pathlib import Path
from tabulate import tabulate

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ontology_server.core.config import get_config


def load_query_config() -> dict:
    """Load query configuration from queries/sparql/config.yaml."""
    config_path = Path(__file__).parent.parent / "queries" / "sparql" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Query config not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def read_query_file(query_file: str) -> str:
    """Read SPARQL query from file."""
    query_path = Path(query_file)

    if not query_path.exists():
        raise FileNotFoundError(f"Query file not found: {query_file}")

    with open(query_path, 'r') as f:
        query = f.read().strip()

    if not query:
        raise ValueError("Query file is empty")

    return query


def execute_query(api_url: str, sparql_query: str) -> dict:
    """Execute SPARQL query via API."""
    try:
        response = requests.post(
            f"{api_url}/sparql",
            json={"query": sparql_query},
            timeout=30
        )

        if response.status_code == 200:
            return response.json()
        else:
            error_detail = response.json().get("detail", "Unknown error")
            raise Exception(f"API Error: {error_detail}")

    except requests.exceptions.ConnectionError:
        raise Exception(f"Cannot connect to server at {api_url}. Make sure the server is running.")
    except Exception as e:
        raise Exception(f"Query execution failed: {e}")


def print_results(results: list, query: str):
    """Print query results in a nice table format."""
    print("🔍 SPARQL Query:")
    print("-" * 60)
    print(query)
    print("-" * 60)
    print()

    if not results:
        print("✓ Query executed successfully")
        print("📊 No results found")
        return

    print("✓ Query executed successfully")
    print(f"📊 Found {len(results)} result(s)")
    print()

    # Format results for table
    formatted_results = []
    for row in results:
        formatted_row = []
        for item in row:
            formatted_row.append(item['value'])
        formatted_results.append(formatted_row)

    # Generate headers
    headers = [f"?var{i+1}" for i in range(len(results[0]))]

    print(tabulate(formatted_results, headers=headers, tablefmt="grid"))


def main():
    """Main function."""
    # Load configuration
    config = get_config()
    server_config = config.get_server_config()
    api_url = f"http://localhost:{server_config['port']}"

    print("=" * 60)
    print("🔍 SPARQL Query Tool")
    print("=" * 60)
    print()

    try:
        # Load query config
        query_config = load_query_config()
        active_query = query_config.get('active_query')

        if not active_query:
            print("✗ No active_query specified in queries/sparql/config.yaml")
            sys.exit(1)

        # Build query file path
        query_file = Path("queries") / "sparql" / active_query

        # Check if query file exists
        if not query_file.exists():
            print(f"✗ Query file not found: {query_file}")
            print()
            print("Available queries should be in: queries/sparql/")
            print(f"Current active_query setting: {active_query}")
            sys.exit(1)

        # Read query from file
        print(f"📄 Active query: {active_query}")
        print(f"📄 Reading from: {query_file}")
        query = read_query_file(str(query_file))
        print()

        # Execute query
        result = execute_query(api_url, query)

        # Print results
        print_results(result['results'], query)

    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
