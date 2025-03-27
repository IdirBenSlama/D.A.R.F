#!/usr/bin/env python3
"""
Script to check Flask installation and dependencies
"""

import os
import sys
import importlib
import subprocess
import pkg_resources

def check_import(module_name):
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✓ Module '{module_name}' successfully imported")
        return True
    except ImportError as e:
        print(f"✗ Failed to import '{module_name}': {e}")
        return False

def check_package_versions():
    """Check installed package versions."""
    print("\nInstalled Package Versions:")
    required_packages = ["flask", "jinja2", "werkzeug", "itsdangerous", "click"]
    
    for package in required_packages:
        try:
            version = pkg_resources.get_distribution(package).version
            print(f"✓ {package}: v{version}")
        except pkg_resources.DistributionNotFound:
            print(f"✗ {package}: Not found")

def check_environment():
    """Check Python environment."""
    print("\nPython Environment:")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.path}")
    
    # Check virtual environment
    if hasattr(sys, 'prefix') and hasattr(sys, 'base_prefix'):
        if sys.prefix != sys.base_prefix:
            print(f"Running in virtual environment: {sys.prefix}")
        else:
            print("Not running in a virtual environment")

def main():
    """Main function."""
    print("Flask Installation Diagnostic Tool")
    print("=================================\n")
    
    # Check Flask import
    print("Import Checks:")
    flask_available = check_import("flask")
    
    if flask_available:
        import flask
        print(f"Flask version: {flask.__version__}")
        print(f"Flask location: {flask.__file__}")
        
    # Check package versions
    check_package_versions()
    
    # Check environment
    check_environment()
    
    # Print recommendations
    print("\nRecommendations:")
    if not flask_available:
        print("- Install Flask: pip install flask")
        print("- Check Python environment: Make sure you're using the same environment across all scripts")
        print("- Try reinstalling requirements: pip install --force-reinstall -r darf_webapp/requirements.txt")
    else:
        print("- Flask is properly installed but might be unavailable in the web application's environment")
        print("- Check for environment issues in the darf_webapp directory")
        print("- Consider creating a symbolic link or copying the Flask package to the web application directory")

if __name__ == "__main__":
    main()
