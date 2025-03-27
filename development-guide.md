# DARF Framework Development Guide

## Overview

This guide provides information for developers who want to contribute to the DARF Framework or extend it for their own use cases. It covers development setup, coding standards, testing practices, and contribution guidelines.

## Development Environment Setup

### Prerequisites

- Python 3.8+
- Node.js 16+ and npm
- Git

### Setting Up Your Development Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/darf-framework.git
   cd darf-framework
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install JavaScript dependencies:
   ```bash
   npm install
   ```

4. Set up ESLint for code quality:
   ```bash
   ./setup_eslint.sh
   ```

### Running the Framework in Development Mode

For development, you can run different components separately:

1. Run the backend only:
   ```bash
   python run.py --backend --debug
   ```

2. Run the frontend only (in a separate terminal):
   ```bash
   python run.py --frontend
   ```

3. Run in debug mode (all components with enhanced logging):
   ```bash
   python run.py --mode debug
   ```

## Project Structure

The DARF Framework is organized into several key directories:

- `darf_frontend/`: TypeScript/React frontend application
- `darf_webapp/`: Python/Flask backend application
- `config/`: Configuration files for different modes
- `modes/`: Mode-specific Python modules
- `docs/`: Documentation files
- `logs/`: Log files generated during runtime
- `tests/`: Automated tests

## Coding Standards

### Python Code Style

- Follow PEP 8 and PEP 257 for code style and docstrings
- Use type hints (PEP 484) for function parameters and return values
- Organize imports alphabetically: standard library, third-party, then local
- Maximum line length: 100 characters
- Use descriptive variable and function names

Example:
```python
from typing import Dict, List, Optional
import json
import os

import flask
import requests

from darf_core import DARFCore


def process_data(input_data: Dict[str, any], config: Optional[Dict[str, any]] = None) -> List[Dict]:
    """
    Process the input data according to the configuration.
    
    Args:
        input_data: The data to process
        config: Optional configuration dictionary
        
    Returns:
        A list of processed data items
    
    Raises:
        ValueError: If input_data is invalid
    """
    if not input_data:
        raise ValueError("Input data cannot be empty")
        
    # Implementation here
    
    return processed_items
```

### JavaScript/TypeScript Code Style

- Follow the Airbnb style guide (enforced via ESLint)
- Use TypeScript for type safety
- Prefer functional components with hooks for React
- Use async/await for asynchronous code
- Use named exports for clarity

Example:
```typescript
import React, { useState, useEffect } from 'react';
import { fetchData } from 'api/dataService';
import { DataItem } from 'types/dataTypes';

interface DataListProps {
  initialFilter?: string;
  maxItems: number;
}

export const DataList: React.FC<DataListProps> = ({ initialFilter = '', maxItems }) => {
  const [data, setData] = useState<DataItem[]>([]);
  const [filter, setFilter] = useState(initialFilter);
  const [loading, setLoading] = useState(false);
  
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      try {
        const result = await fetchData(filter);
        setData(result.slice(0, maxItems));
      } catch (error) {
        console.error('Failed to load data:', error);
      } finally {
        setLoading(false);
      }
    };
    
    loadData();
  }, [filter, maxItems]);
  
  // Component implementation
};
```

## Testing

### Python Testing

- Use pytest for Python tests
- Place tests in the `tests/` directory with a structure mirroring the source
- Aim for high test coverage, especially for core components
- Use fixtures for test data and configuration
- Mock external dependencies

Example:
```python
import pytest
from unittest.mock import MagicMock, patch

from darf_core import DARFCore

@pytest.fixture
def mock_config():
    return {
        "system": {
            "name": "DARF",
            "version": "1.0.0"
        },
        "modes": {
            "test": {
                "components": ["knowledge_graph"],
                "web_port": 5000
            }
        }
    }

@pytest.fixture
def darf_core(mock_config):
    return DARFCore(mock_config)

def test_darf_core_initialization(darf_core, mock_config):
    assert darf_core.config == mock_config
    assert darf_core.components == {}

@patch('darf_core.KnowledgeGraph')
def test_darf_core_load_component(mock_kg_class, darf_core):
    mock_kg = MagicMock()
    mock_kg_class.return_value = mock_kg
    
    darf_core.load_component("knowledge_graph")
    
    assert "knowledge_graph" in darf_core.components
    assert darf_core.components["knowledge_graph"] == mock_kg
    mock_kg_class.assert_called_once_with(darf_core.config)
```

### Running Python Tests

To run the Python tests:

```bash
# Run all tests
python -m pytest tests/

# Run tests with coverage reporting
python -m pytest --cov=src tests/

# Run a specific test file
python -m pytest tests/test_component_interface.py

# Run a specific test class
python -m pytest tests/test_component_interface.py::TestInterfaceValidation

# Run a specific test method
python -m pytest tests/test_component_interface.py::TestInterfaceValidation::test_correct_component
```

For continuous testing during development:

```bash
# Install pytest-watch
pip install pytest-watch

# Watch for changes and run tests automatically
ptw tests/
```

For more detailed testing information, see the [Testing Guide](tests/README.md).

### JavaScript/TypeScript Testing

- Use Jest for unit tests
- Use React Testing Library for component tests
- Place tests next to the source files with a `.test.ts` or `.test.tsx` extension
- Test component behavior, not implementation details

Example:
```typescript
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { DataList } from './DataList';
import { fetchData } from 'api/dataService';

// Mock the API service
jest.mock('api/dataService');

describe('DataList Component', () => {
  const mockFetchData = fetchData as jest.MockedFunction<typeof fetchData>;
  
  beforeEach(() => {
    mockFetchData.mockReset();
  });
  
  it('renders loading state initially', () => {
    mockFetchData.mockImplementation(() => new Promise(resolve => setTimeout(resolve, 100)));
    
    render(<DataList maxItems={5} />);
    
    expect(screen.getByText('Loading...')).toBeInTheDocument();
  });
  
  it('renders data items when loaded', async () => {
    mockFetchData.mockResolvedValue([
      { id: '1', name: 'Item 1' },
      { id: '2', name: 'Item 2' }
    ]);
    
    render(<DataList maxItems={5} />);
    
    await waitFor(() => {
      expect(screen.getByText('Item 1')).toBeInTheDocument();
      expect(screen.getByText('Item 2')).toBeInTheDocument();
    });
  });
  
  it('limits the number of items shown', async () => {
    mockFetchData.mockResolvedValue([
      { id: '1', name: 'Item 1' },
      { id: '2', name: 'Item 2' },
      { id: '3', name: 'Item 3' }
    ]);
    
    render(<DataList maxItems={2} />);
    
    await waitFor(() => {
      expect(screen.getByText('Item 1')).toBeInTheDocument();
      expect(screen.getByText('Item 2')).toBeInTheDocument();
      expect(screen.queryByText('Item 3')).not.toBeInTheDocument();
    });
  });
});
```

### Running JavaScript/TypeScript Tests

To run the frontend tests:

```bash
# Change to the frontend directory
cd darf_frontend

# Run all tests
npm test

# Run tests in watch mode
npm test -- --watch

# Run tests with coverage
npm test -- --coverage
```

## Building and Packaging

### Building the Frontend

```bash
cd darf_frontend
npm run build
```

This will create optimized production files in the `darf_frontend/build` directory.

### Creating a Python Package

```bash
python setup.py sdist
```

This will create a source distribution in the `dist` directory.

## Documentation

All code should be well-documented:

- Python: Use docstrings for modules, classes, and functions
- JavaScript/TypeScript: Use JSDoc comments for functions and components
- Update markdown documentation when changing functionality

## Debug and Troubleshooting

### Debugging the Backend

- Run with debug mode: `python run.py --backend --debug`
- Check logs in the `logs/` directory
- Use Python's debugger (pdb) or an IDE debugger

### Debugging the Frontend

- Use browser developer tools
- Use React Developer Tools extension
- Check the console for errors
- Use `console.log()` statements or debugger breakpoints

## Common Development Tasks

### Adding a New Component

1. Create your component in the appropriate directory
2. Update configuration to include your component
3. Register the component in the component loading system
4. Add tests for your component
5. Update documentation

### Modifying the Configuration System

1. Update the `validate_config` function in `run.py`
2. Add new default values if needed
3. Update configuration documentation
4. Add tests for your changes

### Adding a New API Endpoint

1. Create your endpoint in `darf_webapp/routes.py` or a new routes file
2. Register the endpoint with the Flask app
3. Add authentication if required
4. Add tests for your endpoint
5. Update API documentation

### Extending the Frontend

1. Create your new components in `darf_frontend/src/components`
2. Add to the appropriate pages
3. Update routing if needed
4. Add tests for your components
5. Update frontend documentation

## Contribution Workflow

1. Fork the repository
2. Create a branch for your feature or bugfix
3. Implement your changes with tests
4. Ensure all tests pass: `python -m pytest tests/`
5. Lint your code: `npm run lint`
6. Submit a pull request with a clear description of your changes

## Release Process

1. Update version numbers in `package.json` and Python package metadata
2. Update the changelog
3. Create a release branch
4. Run the full test suite
5. Build production assets
6. Tag the release
7. Merge to main branch
8. Create a GitHub release
9. Publish to package repositories if applicable

## Best Practices

- Write tests before or alongside code (TDD/BDD)
- Review code before submission
- Keep changes focused and small
- Document complex algorithms and decisions
- Use descriptive commit messages
- Follow the principle of least surprise
- Maintain backward compatibility when possible

## Support and Community

- File issues on GitHub for bugs or feature requests
- Use pull requests for code contributions
- Join the community chat for discussions
- Check the wiki for additional documentation
