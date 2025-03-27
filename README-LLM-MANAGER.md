# DARF Multi-LLM Manager

## Overview

The Multi-LLM Manager is an enhancement to the DARF API server that enables handling multiple language models simultaneously and intelligently routing tasks to the most appropriate model based on task type, model capabilities, and performance characteristics.

## Core Components

1. **Configuration System** (`config.py`)
   - Centralized configuration management with environment variable overrides
   - Settings loaded from `darf_config.json` with sensible defaults

2. **LLM Registry** (`llm_registry.py`)
   - Central registry for managing multiple LLMs
   - Model selection based on task classification
   - Performance metrics tracking for each model

3. **Advanced Caching** (`advanced_cache.py`)
   - Improved caching system with Redis support
   - Configurable TTL and cache invalidation

4. **Asynchronous Tasks** (`async_tasks.py`)
   - Celery-based asynchronous processing
   - Non-blocking operation for long-running LLM tasks

5. **Authentication & Authorization** (`auth.py`)
   - API token-based authentication
   - Role-based access control

6. **Error Handling** (`error_handlers.py`)
   - Centralized error handling with detailed information
   - Consistent error response format

## Task Classification and Model Selection

The system automatically analyzes queries to determine the most appropriate LLM:

- Code-related tasks → Code-specialized models (e.g., qwen2.5-coder)
- Knowledge queries → Models with larger context windows
- Creative tasks → Models optimized for creative generation

## Knowledge Graph Integration

Enhanced LLM interaction with the knowledge graph through:

- Specialized prompts for knowledge graph operations
- Context enrichment with relevant graph data
- Structured query conversion

## Implemented Requirements

- ✅ **More Robust Caching**: Redis support for advanced caching needs
- ✅ **Asynchronous Tasks**: Celery integration for non-blocking operation
- ✅ **Security**: Authentication and authorization for API endpoints 
- ✅ **Configuration**: Centralized configuration in separate files
- ✅ **Testing**: Unit tests for key components
- ✅ **Error Handling**: Improved error reporting
- ✅ **Multi-LLM Management**: Support for managing and selecting from multiple models

## Usage

Configure available models in `darf_config.json` or through environment variables. The system will automatically detect models available through Ollama and select the most appropriate model for each task.

```python
# Example API usage
response = requests.post("http://localhost:5000/api/llm/query", json={
    "prompt": "Write a Python function to calculate fibonacci numbers",
    "task_type": "code",  # Optional hint
    "prefer_low_latency": True  # Optional preference
})
```

The API will intelligently route this to a code-specialized model if available.

