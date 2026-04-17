#!/usr/bin/env python3
"""
Configuration Loader
Load configuration from config.yaml
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """Load and manage configuration."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize config loader."""
        if config_path is None:
            # Default to config.yaml in ontology_server directory
            config_path = Path(__file__).parent.parent / "config.yaml"

        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    @property
    def config(self) -> Dict[str, Any]:
        """Get raw configuration dict."""
        return self._config

    def get_active_env(self) -> Optional[str]:
        """Get active environment ID."""
        return self._config.get('active_env')

    def get_server_config(self) -> Dict[str, Any]:
        """Get server configuration."""
        return self._config.get('server', {
            'host': '0.0.0.0',
            'port': 8000
        })

    def get_neo4j_config(self) -> Dict[str, Any]:
        """Get Neo4j configuration."""
        return self._config.get('neo4j', {
            'uri': 'bolt://127.0.0.1:7687',
            'user': 'neo4j',
            'password': 'skan17370208'
        })

    def get_data_config(self) -> Dict[str, Any]:
        """Get data paths configuration."""
        return self._config.get('data', {
            'root': 'data',
            'ontology': 'data/robot.owl',
            'envs_dir': 'data/envs'
        })

    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding configuration for semantic search."""
        return self._config.get('embedding', {
            'model': 'text-embedding-3-small',
            'dimensions': 512
        })

    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration."""
        return self._config


# Global config loader instance
_config_loader = None


def get_config() -> ConfigLoader:
    """Get global config loader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


if __name__ == "__main__":
    # Test configuration loading
    config = get_config()

    print("=" * 60)
    print("Ontology Server Configuration")
    print("=" * 60)
    print()

    active_env = config.get_active_env()
    if active_env:
        print(f"Active Environment: {active_env}")
    else:
        print("Active Environment: None (schema only)")

    print()

    server_config = config.get_server_config()
    print(f"Server: {server_config['host']}:{server_config['port']}")

    print()

    neo4j_config = config.get_neo4j_config()
    print(f"Neo4j: {neo4j_config['uri']}")

    print()
    print("=" * 60)
