#!/usr/bin/env python
"""
Test runner script for DARF Framework.

This script runs the test suite with the appropriate configuration and generates
coverage reports. It should be used as the primary way to run tests during development.
"""

import os
import subprocess
import argparse
import sys
from datetime import datetime


def run_tests(args):
    """Run the test suite with specified options."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Base command
    cmd = ["pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd.append("-v")
    
    # Add test path if specified
    if args.path:
        cmd.append(args.path)
    
    # Add coverage if requested
    if args.coverage:
        cmd.extend([
            "--cov=src",
            f"--cov-report=html:coverage_reports/htmlcov_{timestamp}",
            f"--cov-report=xml:coverage_reports/coverage_{timestamp}.xml",
            "--cov-report=term"
        ])
    
    # Add specific test markers if requested
    if args.markers:
        for marker in args.markers:
            cmd.append(f"-m {marker}")
    
    # Create coverage reports directory if needed
    if args.coverage and not os.path.exists("coverage_reports"):
        os.makedirs("coverage_reports")
    
    # Print command
    print(f"Running: {' '.join(cmd)}")
    
    # Run the tests
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running tests: {e}")
        return e.returncode


def main():
    """Parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Run DARF Framework tests")
    parser.add_argument("path", nargs="?", help="Path to specific test file or directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase verbosity")
    parser.add_argument("-c", "--coverage", action="store_true", help="Generate coverage reports")
    parser.add_argument("-m", "--markers", nargs="+", help="Run tests with specific markers (e.g., unit, integration, e2e)")
    
    # Add predefined test groups
    group = parser.add_argument_group("test groups")
    group.add_argument("--unit", action="store_true", help="Run unit tests only")
    group.add_argument("--integration", action="store_true", help="Run integration tests only")
    group.add_argument("--e2e", action="store_true", help="Run end-to-end tests only")
    group.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    # Set path based on test group
    if args.unit:
        args.path = "tests/unit/"
    elif args.integration:
        args.path = "tests/integration/"
    elif args.e2e:
        args.path = "tests/e2e/"
    elif args.all or (not args.path and not args.unit and not args.integration and not args.e2e):
        args.path = "tests/"
    
    # Run the tests
    return run_tests(args)


if __name__ == "__main__":
    sys.exit(main())
