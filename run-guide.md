# DARF Framework Running Guide

## Overview

The DARF Framework can be run in several different modes depending on your needs. This guide explains the available options and provides examples for running the framework in various configurations.

## Basic Usage

The framework is started using the `run.py` script:

```
python run.py [options]
```

By default, this will run the framework in standard mode with all components.

## Available Modes

The framework supports several operating modes:

| Mode | Description | Command |
|------|-------------|---------|
| Standard | Normal operation with all components | `python run.py` or `python run.py --mode standard` |
| Debug | Additional instrumentation and verbose logging | `python run.py --mode debug` |
| Dashboard | Dashboard-only mode for monitoring | `python run.py --mode dashboard` |
| Minimal | Core functionality only for lightweight operation | `python run.py --mode minimal` |
| Secure | Enhanced security features | `python run.py --mode secure` |
| Frontend-only | Run only the frontend components | `python run.py --frontend` |
| Backend-only | Run only the backend components | `python run.py --backend` |

## Command-Line Options

The following command-line options are available:

| Option | Description | Example |
|--------|-------------|---------|
| `--mode MODE` | Select operating mode (standard, debug, dashboard, minimal, secure) | `python run.py --mode debug` |
| `--config PATH` | Specify custom config file | `python run.py --config custom_config.json` |
| `--port PORT` | Override default port | `python run.py --port 8080` |
| `--no-browser` | Don't automatically open browser | `python run.py --no-browser` |
| `--debug` | Enable Flask debugging (when using --backend) | `python run.py --backend --debug` |
| `--frontend` | Run only the frontend components | `python run.py --frontend` |
| `--backend` | Run only the backend components | `python run.py --backend` |

## Configuration

The framework loads its configuration from the following sources, in order of priority:

1. Custom config file specified with `--config`
2. Mode-specific config file (`config/{mode}.json`)
3. Default config file (`config/default.json`)
4. Hardcoded defaults

Each configuration file should follow the structure outlined in the Configuration Guide.

## Examples

### Running in Standard Mode

```
python run.py
```

This runs the framework in standard mode with all components.

### Running in Debug Mode

```
python run.py --mode debug
```

This runs the framework in debug mode with additional instrumentation and verbose logging.

### Development Setup - Frontend and Backend

For development, it's often useful to run the frontend and backend separately:

1. Terminal 1 - Start the backend:
```
python run.py --backend --debug
```

2. Terminal 2 - Start the frontend:
```
python run.py --frontend
```

The frontend will automatically connect to the backend API through the proxy configuration.

### Testing Changes

You can also use npm scripts to run various components:

```
# Run the whole application
npm start

# Run only the frontend
npm run start:frontend

# Run only the backend
npm run start:backend

# Run in debug mode
npm run start:debug

# Run tests
npm test
```

### Using a Custom Port

```
python run.py --port 8080
```

This runs the framework on port 8080 instead of the default port.

### Using a Custom Configuration File

```
python run.py --config path/to/config.json
```

This runs the framework using a custom configuration file.

## Troubleshooting

If you encounter issues running the framework, check the following:

1. Ensure all dependencies are installed
2. Check the log files in the `logs/` directory for error messages
3. Try running in minimal mode (`python run.py --mode minimal`) to see if basic functionality works
4. Verify your configuration files for syntax errors

If you still have issues, please refer to the troubleshooting guide or contact the development team.
