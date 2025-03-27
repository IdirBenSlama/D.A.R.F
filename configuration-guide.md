# DARF Framework Configuration Guide

## Overview

The DARF Framework uses a flexible configuration system that allows you to customize its behavior for different environments and use cases. This guide explains the configuration system, file formats, and provides examples of common configurations.

## Configuration Hierarchy

The configuration system follows a hierarchy of sources, with each level overriding the previous one:

1. **Minimal Default**: Absolute minimal configuration (only if all other methods fail)
2. **Default Config File**: `config/default.json` for general settings including component definitions
3. **Mode-specific Config**: `config/{mode}.json` for mode-specific settings 
4. **Custom Config File**: Specified via `--config` command-line option
5. **Command-line Parameters**: Override specific settings via command-line

> **Important**: The framework now primarily relies on `config/default.json` rather than hardcoded defaults. Always define your components and modes in this file.

## Default Configuration File

The `config/default.json` file is the central configuration file for the DARF Framework. It contains:

1. **System Information**: Basic framework metadata
2. **Mode Definitions**: All available operational modes and their components
3. **Component Configurations**: Settings for all available components

Example structure of `config/default.json`:

```json
{
  "system": {
    "name": "DARF",
    "version": "1.0.0",
    "description": "Decentralized Autonomous Reaction Framework"
  },
  "modes": {
    "standard": {
      "components": ["knowledge_graph", "event_bus", "llm_interface", "web_ui"],
      "web_port": 5000,
      "description": "Standard mode with all core components"
    },
    "debug": {
      "components": ["knowledge_graph", "event_bus", "llm_interface", "web_ui", "debugger"],
      "debug_level": "verbose",
      "web_port": 5000,
      "description": "Debug mode with additional instrumentation"
    }
  },
  "components": {
    "knowledge_graph": {
      "enabled": true,
      "data_path": "datasets/processed"
    },
    "event_bus": {
      "enabled": true,
      "high_availability": true
    },
    "llm_interface": {
      "enabled": true,
      "model": "default"
    },
    "web_ui": {
      "enabled": true,
      "port": 5000
    },
    "debugger": {
      "enabled": false,
      "level": "standard"
    }
  }
}
```

This file is loaded by default and forms the foundation of your DARF configuration. Mode-specific files and command-line parameters then override these settings as needed.

## Configuration File Format

Configuration files use JSON format with the following structure:

```json
{
  "system": {
    "name": "DARF",
    "version": "1.0.0"
  },
  "modes": {
    "standard": {
      "components": ["knowledge_graph", "event_bus", "llm_interface", "web_ui"],
      "web_port": 5000
    },
    "debug": {
      "components": ["knowledge_graph", "event_bus", "llm_interface", "web_ui", "debugger"],
      "web_port": 5000,
      "debug_level": "verbose"
    }
  },
  "components": {
    "knowledge_graph": {
      "storage_path": "data/knowledge_graph",
      "max_connections": 10
    },
    "event_bus": {
      "queue_size": 1000,
      "persistence": true
    },
    "llm_interface": {
      "default_model": "gpt-4",
      "api_timeout": 30
    },
    "web_ui": {
      "port": 5000,
      "static_path": "darf_webapp/static"
    }
  },
  "logging": {
    "level": "info",
    "file": "logs/darf.log",
    "max_size": 10485760,
    "backup_count": 5
  },
  "security": {
    "enable_authentication": true,
    "session_timeout": 3600,
    "allowed_origins": ["http://localhost:3000"]
  }
}
```

## Required Configuration Sections

### System

Basic system information:

```json
"system": {
  "name": "DARF",
  "version": "1.0.0"
}
```

### Modes

Mode-specific configurations:

```json
"modes": {
  "standard": {
    "components": ["knowledge_graph", "event_bus", "llm_interface", "web_ui"],
    "web_port": 5000
  }
}
```

Each mode must have at least a `components` array listing which components should be active in that mode.

## Component Configuration

Each component can have its own configuration section:

```json
"components": {
  "knowledge_graph": {
    "storage_path": "data/knowledge_graph",
    "max_connections": 10
  }
}
```

## Environment-Specific Configuration

For environment-specific settings (development, staging, production), you can create separate configuration files and use the `--config` option:

```
python run.py --config config/production.json
```

## Frontend Configuration

The frontend (React/TypeScript) can access configuration values through environment variables or a dedicated API endpoint:

### Environment Variables

In `darf_frontend/.env`:

```
REACT_APP_API_URL=http://localhost:5000/api
REACT_APP_DEBUG=false
```

### API Endpoint

The backend provides a `/api/config` endpoint that returns frontend-specific configuration.

## Validation

The configuration system validates the configuration during loading:

- Checking for required fields
- Validating types
- Ensuring mode-specific required components

If validation fails, the system will log detailed error messages and either:
1. Fall back to defaults (if possible)
2. Exit with an error code

## Examples

### Minimal Configuration

```json
{
  "system": {
    "name": "DARF",
    "version": "1.0.0"
  },
  "modes": {
    "minimal": {
      "components": ["knowledge_graph"],
      "web_port": 5000
    }
  }
}
```

### Development Configuration

```json
{
  "system": {
    "name": "DARF",
    "version": "1.0.0"
  },
  "modes": {
    "standard": {
      "components": ["knowledge_graph", "event_bus", "llm_interface", "web_ui"],
      "web_port": 5000
    },
    "debug": {
      "components": ["knowledge_graph", "event_bus", "llm_interface", "web_ui", "debugger"],
      "web_port": 5000,
      "debug_level": "verbose"
    }
  },
  "components": {
    "web_ui": {
      "port": 5000,
      "debug": true,
      "hot_reload": true
    }
  },
  "logging": {
    "level": "debug"
  },
  "security": {
    "enable_authentication": false
  }
}
```

### Production Configuration

```json
{
  "system": {
    "name": "DARF",
    "version": "1.0.0"
  },
  "modes": {
    "standard": {
      "components": ["knowledge_graph", "event_bus", "llm_interface", "web_ui"],
      "web_port": 80
    }
  },
  "components": {
    "web_ui": {
      "port": 80,
      "debug": false,
      "hot_reload": false
    }
  },
  "logging": {
    "level": "info"
  },
  "security": {
    "enable_authentication": true,
    "session_timeout": 1800,
    "allowed_origins": ["https://darf.example.com"]
  }
}
```

## Best Practices

1. **Version Control**: Keep configuration files in version control with environment-specific files ignored
2. **Secrets Management**: Never store secrets (API keys, passwords) in configuration files
3. **Documentation**: Document all configuration options in code and configuration files
4. **Validation**: Always validate configuration at startup to catch errors early
5. **Defaults**: Provide sensible defaults for all configuration options
6. **Overrides**: Use command-line options for temporary overrides, not permanent changes

## Troubleshooting

If the framework doesn't start or behaves unexpectedly, check the following:

1. **Configuration Syntax**: Ensure your JSON is valid (no missing commas, brackets, etc.)
2. **Missing Required Fields**: Check for missing required configuration fields
3. **Type Errors**: Ensure all values have the correct type (number, string, boolean, array)
4. **Path Issues**: Verify that all paths exist and are accessible
5. **Port Conflicts**: Ensure the configured ports are not already in use
6. **Log Output**: Check the logs for detailed error messages about configuration issues
