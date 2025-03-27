#!/usr/bin/env python3
"""
Script to verify the consolidated DARF framework setup.
This script will:
1. Check for required packages
2. Verify key files and modules exist
3. Test that the framework can be imported and initialized
"""

import os
import sys
import importlib
import logging
from datetime import datetime

# Configure logging
log_filename = f"logs/darf_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename)
    ]
)

logger = logging.getLogger("DARF.Verification")

# Required packages
REQUIRED_PACKAGES = [
    "flask",
    "numpy",
    "networkx"
]

# Key files that should exist
KEY_FILES = [
    "DARF.py",
    "darf_core.py",
    "darf_cli.py",
    "darf_config.json",
    "default_config.json",
    "run_integrated_darf.py"
]

# Key modules in src/modules
KEY_MODULES = [
    "consensus_engine",
    "knowledge_graph_engine",
    "fuzzy_logic_engine",
    "resource_governor",
    "vault_module"
]

def check_packages():
    """Check if required packages are installed"""
    logger.info("Checking required packages...")
    
    missing_packages = []
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
            logger.info(f"✓ {package} is installed")
        except ImportError:
            logger.error(f"✗ {package} is not installed")
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Missing packages: {missing_packages}")
        logger.warning("You can install them with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_files():
    """Check if key files exist"""
    logger.info("Checking key files...")
    
    missing_files = []
    for file in KEY_FILES:
        if os.path.exists(file):
            logger.info(f"✓ {file} exists")
        else:
            logger.error(f"✗ {file} is missing")
            missing_files.append(file)
    
    if missing_files:
        logger.warning(f"Missing files: {missing_files}")
        return False
    
    return True

def check_modules():
    """Check if key modules exist"""
    logger.info("Checking key modules...")
    
    if not os.path.exists("src/modules"):
        logger.error("✗ src/modules directory does not exist")
        return False
    
    missing_modules = []
    for module in KEY_MODULES:
        module_path = os.path.join("src", "modules", module)
        if os.path.exists(module_path):
            logger.info(f"✓ {module} module exists")
        else:
            logger.error(f"✗ {module} module is missing")
            missing_modules.append(module)
    
    if missing_modules:
        logger.warning(f"Missing modules: {missing_modules}")
        return False
    
    return True

def test_darf_import():
    """Test importing and initializing DARF"""
    logger.info("Testing DARF import...")
    
    try:
        # Add current directory to path
        if not os.getcwd() in sys.path:
            sys.path.insert(0, os.getcwd())
        
        # Try importing DARF
        import DARF
        logger.info("✓ DARF module imported successfully")
        
        # Create an instance
        logger.info("Testing DARF initialization...")
        try:
            darf = DARF.DARFFramework()
            logger.info("✓ DARF framework initialized successfully")
            return True
        except Exception as e:
            logger.error(f"✗ Error initializing DARF framework: {e}")
            return False
    except Exception as e:
        logger.error(f"✗ Error importing DARF: {e}")
        return False

def main():
    """Main verification function"""
    logger.info("Starting DARF framework verification...")
    
    # Check required packages
    packages_ok = check_packages()
    
    # Check key files
    files_ok = check_files()
    
    # Check key modules
    modules_ok = check_modules()
    
    # Test DARF import
    import_ok = test_darf_import()
    
    # Print summary
    logger.info("\nVerification Summary:")
    logger.info(f"Packages: {'✓ OK' if packages_ok else '✗ FAILED'}")
    logger.info(f"Files: {'✓ OK' if files_ok else '✗ FAILED'}")
    logger.info(f"Modules: {'✓ OK' if modules_ok else '✗ FAILED'}")
    logger.info(f"Import test: {'✓ OK' if import_ok else '✗ FAILED'}")
    
    if packages_ok and files_ok and modules_ok and import_ok:
        logger.info("\n✓ DARF framework verification PASSED")
        logger.info("You can run the framework with: python run_integrated_darf.py")
        return True
    else:
        logger.error("\n✗ DARF framework verification FAILED")
        logger.error("Please fix the issues above before running the framework")
        return False

if __name__ == "__main__":
    main()
