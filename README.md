to simulate # DARF Framework - Decentralized Autonomous Reaction Framework

## Quick Start

Get DARF up and running in just 3 simple steps:

1. **Install Dependencies**:
   ```bash
   # Clone the repository
   git clone https://github.com/your-org/darf-framework.git
   cd darf-framework
   
   # Install dependencies
   pip install -r requirements.txt
   npm install
   ```

2. **Run the Framework**:
   ```bash
   # Start DARF in standard mode
   python run.py
   ```

3. **Access the Interface**:
   Open your browser to http://localhost:5000 to access the DARF web interface.

That's it! For more detailed information, continue reading below.

## Overview

DARF is a comprehensive framework for building decentralized, autonomous systems with reactive capabilities. It provides a modular architecture with components for knowledge graphs, event processing, LLM interfaces, and web UIs.

## Key Features

- **Modular Architecture**: Mix and match components based on your needs
- **Multiple Operation Modes**: Standard, Debug, Dashboard, Minimal, and Secure modes
- **Flexible Configuration**: Configuration through files or command-line arguments
- **TypeScript/JavaScript Support**: Frontend development with modern JS/TS practices
- **Python Backend**: Robust backend with asyncio support

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js 16+ and npm
- Git

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-org/darf-framework.git
   cd darf-framework
   ```

2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install JavaScript dependencies:
   ```
   npm install
   ```

4. Set up ESLint for code quality:
   ```
   ./setup_eslint.sh
   ```

### Running the Framework

The framework can be run in several modes:

```
# Standard mode (default)
python run.py

# Debug mode
python run.py --mode debug

# Dashboard mode
python run.py --mode dashboard

# Minimal mode
python run.py --mode minimal

# Secure mode
python run.py --mode secure
```

Additional options:
```
# Specify custom config file
python run.py --config path/to/config.json

# Specify custom port
python run.py --port 8080

# Run without automatically opening browser
python run.py --no-browser
```

## Architecture

DARF consists of several key components:

- **Knowledge Graph**: Core data structure for storing and relating information
- **Event Bus**: Handles event routing between components
- **LLM Interface**: Interface with Language Model APIs
- **Web UI**: Browser-based interface for interacting with the framework
- **Debugger**: Tools for debugging and monitoring (Debug mode only)

Components are loaded based on the selected operation mode. Configuration files define which components are used in each mode.

## Configuration

Configuration can be loaded from:
1. Custom config file specified with `--config`
2. Mode-specific config file (`config/{mode}.json`)
3. Default config file (`config/default.json`)
4. Hardcoded defaults if no files exist

## Development

### Frontend

The frontend is located in `darf_frontend/` and uses TypeScript with React.

To lint frontend code:
```
npm run lint:frontend
```

### Backend

The backend is primarily Python-based, with the main app in `darf_webapp/`.

To lint backend code:
```
npm run lint:backend
```

## Documentation

- See `eslint-guide.md` for ESLint configuration details
- See `docker_security_implementation_guide.md` for Docker security guidelines
- See `LLM_GRAPH_INTEGRATION_README.md` for LLM integration details

## License

[Add appropriate license information here]
