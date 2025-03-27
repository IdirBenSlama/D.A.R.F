#!/usr/bin/env python3
"""
DARF - Decentralized Autonomous Reaction Framework
Unified Run Script

This is the single entry point for running DARF in all modes:

Usage:
  python run.py [--mode MODE] [--config PATH] [--port PORT] [--no-browser]
  python run.py --frontend  # Run only the frontend
  python run.py --backend   # Run only the backend
  python run.py --help      # Show detailed help

Available modes:
  standard    : Normal operation with all components (default)
  debug       : Debug mode with additional instrumentation
  dashboard   : Dashboard-only mode for monitoring
  minimal     : Core functionality only
  secure      : Enhanced security features
"""

import argparse
import os
import sys
import json
import time
import logging
import asyncio
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional, List, Set

# Import custom errors
from src.errors import DARFError, ConfigurationError, ComponentError, ComponentNotFoundError

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/darf_{time.strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

logger = logging.getLogger("DARF.Runner")

def load_config(config_path: Optional[str] = None, mode: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file and validate using the ConfigManager.
    
    Args:
        config_path: Path to configuration file (optional)
        mode: Operation mode to use if no config file provided (optional)
        
    Returns:
        Dictionary with configuration
    
    Raises:
        ConfigurationError: If configuration cannot be loaded or is invalid
    """
    try:
        # Import the ConfigManager
        from src.config.config_manager import config_manager
        
        # Define minimal default configuration
        minimal_default = {
            "system": {
                "name": "DARF",
                "version": "1.0.0"
            }
        }
        
        # Try to load from specified file first
        if config_path:
            if not os.path.exists(config_path):
                raise ConfigurationError(f"Configuration file not found: {config_path}")
            
            logger.info(f"Loading configuration from {config_path}")
            try:
                config = config_manager.load_configuration(config_path, required=True)
                return config
            except json.JSONDecodeError as e:
                raise ConfigurationError(f"Invalid JSON in configuration file {config_path}: {e}")
            except Exception as e:
                if not isinstance(e, ConfigurationError):
                    e = ConfigurationError(f"Error loading configuration from {config_path}: {e}")
                raise e
        
        # If no specified file, try mode-specific config
        if mode:
            mode_config_path = f"config/{mode}.json"
            if os.path.exists(mode_config_path):
                logger.info(f"Loading {mode} configuration from {mode_config_path}")
                try:
                    config = config_manager.load_configuration(mode_config_path, required=False)
                    if config:
                        return config
                except Exception as e:
                    logger.error(f"Error loading {mode} configuration: {e}")
        
        # Try default config
        default_config_path = "config/default.json"
        if os.path.exists(default_config_path):
            logger.info(f"Loading default configuration from {default_config_path}")
            try:
                config = config_manager.load_configuration(default_config_path, required=False)
                if config:
                    return config
            except Exception as e:
                logger.error(f"Error loading default configuration: {e}")
        
        # If we get here, we need to use minimal default
        logger.warning("No valid configuration found. Using minimal defaults.")
        return minimal_default
    except Exception as e:
        if isinstance(e, ConfigurationError):
            raise e
        logger.error(f"Unexpected error loading configuration: {e}")
        raise ConfigurationError(f"Unexpected error loading configuration: {e}")

def validate_config(config: Dict[str, Any], mode: Optional[str] = None) -> None:
    """
    Validate the configuration using the ConfigManager.
    
    Args:
        config: Configuration dictionary to validate
        mode: Operation mode to validate for (optional)
        
    Raises:
        ConfigurationError: If the configuration is invalid
    """
    from src.config.config_manager import ConfigManager
    
    # Create a temporary config manager to validate the config
    config_manager = ConfigManager()
    
    # Validate the configuration
    validation_errors = config_manager.validate_configuration(config)
    if validation_errors:
        error_str = "\n".join(validation_errors)
        logger.error(f"Configuration validation failed:\n{error_str}")
        raise ConfigurationError(f"Configuration validation failed:\n{error_str}")
    
    # Validate mode-specific configuration if a mode is specified
    if mode:
        # First set the config in the manager
        config_manager.config = config
        
        # Then validate the mode
        mode_errors = config_manager.validate_mode(mode)
        if mode_errors:
            error_str = "\n".join(mode_errors)
            logger.error(f"Mode validation failed for '{mode}':\n{error_str}")
            raise ConfigurationError(f"Mode validation failed for '{mode}':\n{error_str}")

def run_mode(mode_name: str, args: argparse.Namespace, config: Dict[str, Any]) -> int:
    """
    Run a specific mode with proper error handling.
    
    Args:
        mode_name: Name of the mode to run
        args: Command line arguments
        config: Configuration dictionary
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    logger.info(f"Starting DARF in {mode_name} mode")
    
    # Verify mode exists in configuration
    if "modes" not in config or mode_name not in config["modes"]:
        logger.error(f"Mode '{mode_name}' not defined in configuration")
        return 1
        
    # Special case handlers
    if mode_name == "minimal":
        return run_minimal_mode(args, config)
    elif mode_name == "frontend":
        return run_frontend(args, config)
    elif mode_name == "backend":
        return run_backend(args, config)
    
    # For other modes, try to import the mode module
    try:
        # Try to import from modes package
        module_path = f"modes.{mode_name}_mode"
        try:
            logger.debug(f"Attempting to import mode module from package: {module_path}")
            module = importlib.import_module(module_path)
        except ImportError as import_err:
            logger.debug(f"Failed to import from package: {str(import_err)}")
            
            # If the module doesn't exist in the package, check if there's a file we can load directly
            module_file = f"modes/{mode_name}_mode.py"
            if not os.path.exists(module_file):
                logger.error(f"Mode module '{mode_name}' not found. "
                           f"Searched in package 'modes.{mode_name}_mode' and "
                           f"file '{module_file}'.")
                logger.info("Available modes: standard, debug, dashboard, minimal, secure, frontend, backend")
                logger.info("Example: python run.py --mode=standard")
                return 1
            
            # Dynamically load the module from the file
            logger.debug(f"Attempting to load module from file: {module_file}")
            spec = importlib.util.spec_from_file_location(f"{mode_name}_mode", module_file)
            if spec is None or spec.loader is None:
                logger.error(f"Failed to create module spec for: {module_file}")
                return 1
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        
        # Check if the module has a run function
        if not hasattr(module, "run"):
            logger.error(f"Module '{module_path}' does not have a 'run' function. "
                        f"Each mode module must implement a run(args, config) function.")
            return 1
        
        # Run the mode
        logger.debug(f"Executing run function from mode module: {mode_name}")
        return module.run(args, config)
        
    except ImportError as e:
        logger.error(f"Failed to import dependencies for {mode_name} mode: {e}")
        logger.info("Make sure all required packages are installed: pip install -r requirements.txt")
        return 1
    except TypeError as e:
        logger.error(f"Type error in {mode_name} mode: {e}")
        logger.info("This could be due to incompatible function arguments or return values.")
        return 1
    except FileNotFoundError as e:
        logger.error(f"File not found in {mode_name} mode: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error running {mode_name} mode: {e}", exc_info=True)
        return 1

def run_minimal_mode(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    """Run DARF in minimal mode with strict component validation."""
    logger.info("Starting DARF in minimal mode")
    
    try:
        # Import core components
        from darf_class import DARF
        from darf_core import DARFCore
        from src.config.config_manager import ConfigManager
        
        # Validate mode configuration
        config_manager = ConfigManager()
        config_manager.config = config
        errors = config_manager.validate_mode("minimal")
        if errors:
            error_str = "\n".join(errors)
            logger.error(f"Invalid minimal mode configuration:\n{error_str}")
            return 1
        
        # Extract mode configuration from config
        mode_config = config.get("modes", {}).get("minimal", {})
        components = mode_config.get("components", ["knowledge_graph"])
        
        # Check that all components are defined in the components section
        missing_components = []
        for component_name in components:
            if component_name not in config.get("components", {}):
                missing_components.append(component_name)
                
        if missing_components:
            missing_str = ", ".join(missing_components)
            logger.error(f"Components referenced in minimal mode are not defined: {missing_str}")
            return 1
        
        logger.info(f"Configured components for minimal mode: {', '.join(components)}")
        
        # Create DARF instance with specified components
        core_config = config.copy()
        core_config["active_components"] = components
        
        core = DARFCore(core_config)
        darf = DARF(core=core)
        
        # Run minimal DARF
        asyncio.run(darf.start())
        
        # Keep running until interrupted
        logger.info("DARF minimal mode is running. Press Ctrl+C to exit.")
        try:
            asyncio.run(asyncio.sleep(float('inf')))
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            asyncio.run(darf.shutdown())
        
        return 0
    except ImportError as e:
        logger.error(f"Failed to import minimal mode components: {e}")
        return 1
    except ComponentError as e:
        logger.error(f"Component error: {e}")
        return 1
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error running minimal mode: {e}", exc_info=True)
        return 1

def run_frontend(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    """Run only the frontend."""
    logger.info("Starting DARF frontend")
    
    try:
        # Get the frontend port from config or default to 3000
        port = 3000
        if "modes" in config and "frontend" in config["modes"]:
            if "web_port" in config["modes"]["frontend"]:
                port = config["modes"]["frontend"]["web_port"]
        
        if args.port:
            port = args.port
        
        # Run the frontend with npm
        import subprocess
        
        # Check if the frontend directory exists
        if not os.path.exists("darf_frontend"):
            logger.error("Frontend directory 'darf_frontend' not found")
            return 1
        
        # Check for package.json existence
        if not os.path.exists("darf_frontend/package.json"):
            logger.error("package.json not found in frontend directory")
            logger.info("Make sure the frontend is properly set up")
            return 1
            
        # Change to the frontend directory and run npm start
        logger.info(f"Starting frontend on port {port}")
        
        # Set the PORT environment variable
        env = os.environ.copy()
        env["PORT"] = str(port)
        
        # Use npm start to run the frontend with better error handling
        try:
            process = subprocess.Popen(
                ["npm", "start"],
                cwd="darf_frontend",
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for the process to finish
            exit_code = process.wait()
            
            # Check if there was an error
            if exit_code != 0:
                stdout, stderr = process.communicate()
                logger.error(f"Frontend process exited with code {exit_code}")
                logger.error(f"Frontend stdout: {stdout}")
                logger.error(f"Frontend stderr: {stderr}")
                return exit_code
            
            return 0
            
        except FileNotFoundError:
            logger.error("npm command not found. Make sure Node.js is installed.")
            return 1
        except PermissionError:
            logger.error("Permission denied when trying to run npm start.")
            return 1
        except subprocess.SubprocessError as e:
            logger.error(f"Subprocess error when running frontend: {e}")
            return 1
        
    except ImportError as e:
        logger.error(f"Failed to import required module: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error running frontend: {e}", exc_info=True)
        return 1

def run_backend(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    """Run only the backend."""
    logger.info("Starting DARF backend")
    
    try:
        # Get the backend port from config or default to 5000
        port = 5000
        if "modes" in config and "backend" in config["modes"]:
            if "web_port" in config["modes"]["backend"]:
                port = config["modes"]["backend"]["web_port"]
        
        if args.port:
            port = args.port
        
        # Check if webapp directory exists
        if not os.path.exists("darf_webapp"):
            logger.error("Backend directory 'darf_webapp' not found")
            return 1
            
        # Check for app.py existence
        if not os.path.exists("darf_webapp/app.py"):
            logger.error("app.py not found in backend directory")
            logger.info("Make sure the backend is properly set up")
            return 1
        
        try:
            # Import the backend app
            from darf_webapp.app import app
            
            # Try to check if the port is available
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.bind(('0.0.0.0', port))
            except socket.error as e:
                logger.error(f"Port {port} is already in use or not available: {e}")
                logger.info(f"Try a different port with --port option")
                return 1
            finally:
                s.close()
            
            # Run the Flask app
            logger.info(f"Starting backend on port {port}")
            app.run(host="0.0.0.0", port=port, debug=args.debug)
            
            return 0
            
        except ImportError as e:
            logger.error(f"Failed to import backend components: {e}")
            logger.error(f"Make sure Flask is installed: pip install flask")
            return 1
        except OSError as e:
            logger.error(f"OS error when starting backend: {e}")
            logger.error(f"This might be due to port {port} being in use. Try a different port.")
            return 1
        except ModuleNotFoundError as e:
            logger.error(f"Module not found: {e}")
            logger.error("Make sure all required packages are installed: pip install -r requirements.txt")
            return 1
        
    except Exception as e:
        logger.error(f"Error running backend: {e}", exc_info=True)
        return 1

def main() -> int:
    """
    Main entry point.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="DARF Unified Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                   # Run in standard mode
  python run.py --mode debug      # Run in debug mode
  python run.py --frontend        # Run only the frontend
  python run.py --backend         # Run only the backend
  python run.py --port 8080       # Run with a custom port
  python run.py --config custom.json  # Run with a custom config
        """
    )
    
    # Mode group
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--mode", 
        choices=["standard", "debug", "dashboard", "minimal", "secure"], 
        default="standard", 
        help="Operation mode (default: standard)"
    )
    mode_group.add_argument(
        "--frontend",
        action="store_true",
        help="Run only the frontend"
    )
    mode_group.add_argument(
        "--backend",
        action="store_true",
        help="Run only the backend"
    )
    
    # Other arguments
    parser.add_argument(
        "--config",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically"
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Server port (overrides config)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for Flask"
    )
    
    args = parser.parse_args()
    
    # Determine the actual mode to run
    mode = args.mode
    if args.frontend:
        mode = "frontend"
    elif args.backend:
        mode = "backend"
    
    # Print welcome message
    print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   DARF - Decentralized Autonomous Reaction Framework              ║
║                                                                   ║
║   Mode: {mode:<56} ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Load configuration
        config = load_config(args.config, mode)
        
        # Additional validation
        validate_config(config, mode)
        
        # Update config with command-line overrides
        if args.port:
            if "modes" in config and mode in config["modes"]:
                config["modes"][mode]["web_port"] = args.port
            
            # Also update component config if it exists
            if "components" in config and "web_ui" in config["components"]:
                config["components"]["web_ui"]["port"] = args.port
        
        # Initialize asyncio policy for Windows if needed
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # Run the selected mode
        return run_mode(mode, args, config)
        
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except DARFError as e:
        logger.error(f"DARF error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)
