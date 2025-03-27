"""Configuration module for the DARF API server."""
import os
import json
import logging

logger = logging.getLogger(__name__)

class Config:
    def __init__(self, config_file='darf_config.json'):
        self.config_file = config_file
        self._config = {}
        self._load_config()
        
    def _load_config(self):
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self._config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_file}")
            else:
                logger.warning(f"Configuration file {self.config_file} not found. Using defaults.")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    def __getattr__(self, name):
        # Check environment variable
        env_var = os.environ.get(name.upper())
        if env_var is not None:
            # Convert to appropriate type
            if env_var.lower() in ('true', 'yes', '1'):
                return True
            elif env_var.lower() in ('false', 'no', '0'):
                return False
            try:
                if '.' in env_var:
                    return float(env_var)
                return int(env_var)
            except ValueError:
                return env_var
        
        # Check configuration file
        if name in self._config:
            return self._config[name]
        
        # Return default if not found
        return self._get_default(name)
    
    def _get_default(self, name):
        defaults = {
            "PORT": 5000,
            "HOST": "0.0.0.0",
            "DEBUG": False,
            "LOG_LEVEL": "INFO",
            "LOG_FILE": "logs/darf_api.log",
            "CACHE_ENABLED": True,
            "CACHE_TYPE": "memory",
            "CACHE_TTL": 300,
            "AUTH_ENABLED": False,
            "API_TOKENS_FILE": "api_tokens.json",
            "PROMETHEUS_URL": "http://localhost:9090",
            "DEFAULT_LLM_MODEL": "qwen2.5-coder:14b",
            "OLLAMA_API_URL": "http://localhost:11434",
            "MAX_CONVERSATION_HISTORY": 10,
            "ASYNC_ENABLED": False
        }
        return defaults.get(name)

# Create a global instance of the configuration
current_config = Config()
