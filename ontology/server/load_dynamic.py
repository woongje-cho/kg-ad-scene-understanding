#!/usr/bin/env python3
"""
Dynamic Instance Loader (TTL)
Load dynamic instances from TTL file and send to ontology manager
Reads active space from config.yaml
"""

import requests
from pathlib import Path
from typing import Dict, Any
import sys


class DynamicLoader:
    """Load dynamic instances from TTL file."""

    def __init__(self, api_url: str = "http://localhost:8000"):
        """Initialize dynamic loader."""
        self.api_url = api_url

    def load_from_ttl(self, ttl_path: str) -> Dict[str, Any]:
        """
        Load dynamic instances from TTL file via API.

        Args:
            ttl_path: Path to TTL file containing individuals

        Returns:
            Status dictionary with load results
        """
        try:
            # Check if file exists
            if not Path(ttl_path).exists():
                print(f"✗ File not found: {ttl_path}")
                return {"status": "error", "message": "File not found"}

            print(f"📖 Loading dynamic instances from TTL: {ttl_path}")

            # Send to API
            print(f"🚀 Sending request to server...")

            try:
                response = requests.post(
                    f"{self.api_url}/load_ttl",
                    json={"file_path": str(ttl_path)},
                    timeout=12000
                )

                if response.status_code == 200:
                    result = response.json()
                    added_count = result.get("added", 0)
                    failed_count = result.get("failed", 0)

                    print(f"\n✓ Successfully loaded {added_count} individuals")
                    if failed_count > 0:
                        print(f"⚠ Failed to load {failed_count} individuals")
                    print(f"📊 Status: {result.get('status', 'unknown')}")
                else:
                    error_detail = response.json().get("detail", "Unknown error")
                    print(f"✗ Request failed: {error_detail}")
                    return {"status": "error", "message": error_detail}

            except Exception as e:
                print(f"✗ Error in request: {e}")
                return {"status": "error", "message": str(e)}

            return {
                "status": "success",
                "added": added_count,
                "failed": failed_count
            }

        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            return {"status": "error", "message": str(e)}

    def check_server(self) -> bool:
        """Check if ontology manager server is running."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✓ Server is running at {self.api_url}")
                return True
            else:
                print(f"✗ Server returned status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"✗ Cannot connect to server at {self.api_url}")
            print("  Make sure the server is running: python cli/run_server.py")
            return False
        except Exception as e:
            print(f"✗ Error checking server: {e}")
            return False


def main():
    """Main function for command-line usage."""
    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from ontology_server.core.config import get_config
    from ontology_server.core.env import EnvManager

    # Load configuration
    config = get_config()
    active_env = config.get_active_env()
    server_config = config.get_server_config()

    api_url = f"http://localhost:{server_config['port']}"

    if not active_env:
        print("✗ No active space configured in config.yaml")
        print("  Please set 'active_env' in ontology_server/config.yaml")
        sys.exit(1)

    # Get space file path
    space_manager = EnvManager()
    ttl_path = space_manager.get_dynamic_file_path(active_env)

    if not ttl_path:
        print(f"✗ Environment '{active_env}' not found or has no dynamic.ttl file")
        sys.exit(1)

    print("🚀 Dynamic Instance Loader")
    print("=" * 50)
    print(f"📍 Active Environment: {active_env}")
    print()

    loader = DynamicLoader(api_url=api_url)

    # Check server
    if not loader.check_server():
        sys.exit(1)

    # Load instances from TTL
    result = loader.load_from_ttl(str(ttl_path))

    if result["status"] == "error":
        sys.exit(1)

    print("\n✅ Dynamic instances loaded successfully!")


if __name__ == "__main__":
    main()
