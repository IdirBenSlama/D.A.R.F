#!/usr/bin/env python3
"""
DARF Framework Initialization Script

This script initializes the DARF Framework with improved configuration validation
and component management. It:
1. Sets up the required directory structure
2. Creates configuration files
3. Initializes the basic components
4. Validates the configuration and component interfaces
"""

import os
import sys
import logging
import asyncio
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"darf_init_{__import__('time').strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

logger = logging.getLogger("DARF.Initializer")

async def initialize_darf(args):
    """
    Initialize the DARF Framework.
    
    Args:
        args: Command-line arguments
    
    Returns:
        Success status
    """
    logger.info("Initializing DARF Framework with improved configuration and component management")
    
    try:
        # Import initialization utilities
        try:
            from src.utils.initialize import initialize_framework
        except ImportError:
            logger.info("Creating initial directory structure...")
            # Create src directory and utils subdirectory if they don't exist
            os.makedirs("src/utils", exist_ok=True)
            
            # If initialize.py doesn't exist yet, we can't import it
            logger.error("Please run this script after downloading the framework files")
            return False
        
        # Initialize the framework directory structure and configuration
        initialize_framework()
        
        # Import and validate component interfaces
        try:
            from src.interfaces.component import Component
            from src.types.common_types import Result
            from src.types.component_types import ComponentStatus
            from src.config.config_manager import config_manager
            
            logger.info("Successfully imported core modules")
        except ImportError as e:
            logger.error(f"Error importing core modules: {e}")
            logger.info("Please ensure all framework files are in place")
            return False
        
        # Load and validate configuration
        try:
            # Try to load default configuration
            config = config_manager.load_configuration("config/default.json")
            logger.info("Default configuration loaded and validated successfully")
            
            # Print configured components
            if "components" in config:
                components = list(config["components"].keys())
                logger.info(f"Configured components: {', '.join(components)}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return False
        
        # Try to initialize the Knowledge Graph component
        try:
            from src.modules.knowledge_graph import KnowledgeGraphComponent
            
            # Create a component instance
            component = KnowledgeGraphComponent(
                "knowledge_graph",
                config.get("components", {}).get("knowledge_graph", {})
            )
            
            # Start the component
            success = await component.start()
            if success:
                logger.info("Knowledge Graph component started successfully")
                
                # Get component status
                status = component.get_status()
                logger.info(f"Component status: {status}")
                
                # Stop the component
                await component.stop()
                logger.info("Knowledge Graph component stopped successfully")
            else:
                logger.error("Failed to start Knowledge Graph component")
                return False
        except ImportError as e:
            logger.error(f"Error importing Knowledge Graph component: {e}")
            logger.info("Make sure the Knowledge Graph component is properly implemented")
            return False
        except Exception as e:
            logger.error(f"Error initializing Knowledge Graph component: {e}")
            return False
        
        logger.info("DARF Framework initialization completed successfully")
        return True
    except Exception as e:
        logger.error(f"Unexpected error during initialization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """
    Main entry point.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="DARF Framework Initializer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dir",
        help="Base directory for initialization (default: current directory)"
    )
    
    parser.add_argument(
        "--config",
        help="Path to custom configuration file"
    )
    
    args = parser.parse_args()
    
    # Print welcome message
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   DARF - Decentralized Autonomous Reaction Framework              ║
║   Enhanced Configuration and Component Management                 ║
║                                                                   ║
║   Initializing framework...                                       ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize asyncio policy for Windows if needed
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Run the initialization
    success = asyncio.run(initialize_darf(args))
    
    if success:
        print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   DARF Framework initialized successfully!                        ║
║                                                                   ║
║   To run the framework:                                           ║
║     python run.py                                                 ║
║                                                                   ║
║   For more options:                                               ║
║     python run.py --help                                          ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
        """)
        return 0
    else:
        print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   DARF Framework initialization failed.                           ║
║                                                                   ║
║   Please check the log file for details.                          ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
        """)
        return 1

if __name__ == "__main__":
    sys.exit(main())
