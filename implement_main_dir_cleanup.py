#!/usr/bin/env python3
"""
DARF Framework Directory Cleanup Implementation

This script implements the cleanup plan to remove redundant and unused files
from the DARF Framework repository based on the cleanup analysis.

Usage:
    python implement_main_dir_cleanup.py [--dry-run]
"""

import os
import sys
import shutil
import argparse
import json
import datetime
import logging
from pathlib import Path

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"cleanup_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file),
    ],
)
logger = logging.getLogger("DARF.Cleanup")

# Lists of files and directories to remove
REDUNDANT_RUN_SCRIPTS = [
    "run_darf_datasets.py",  # Duplicate identified in cleanup plan
    "run_darf_api.py",        # Functionality covered in run_darf_secured.py
    "run_darf_dashboard_only.py",  # Limited functionality covered by other scripts
    "run_darf_frontend.py",   # Functionality covered in run_darf_secured.py
    "run_darf_minimal.py",    # Basic functionality covered by other scripts
    "run_darf_simple.py",     # Simple version superseded by more complete versions
    "run_darf_enhanced.py",   # Enhanced functionality now in consolidated versions
    "run_darf_prometheus_grafana.py",  # Functionality now in secured version
    "run_darf_integrated.py", # Integrated functionality now in consolidated versions
    "run_darf_integrated_lifecycle.py", # Functionality now in consolidated versions
    "run_darf_consolidated.py", # Now using run_darf_all_in_one.py
]

REDUNDANT_SCRIPT_FILES = [
    "darf_api_server.py.bak",  # Backup file no longer needed
    "debug_darf_system.py",    # Debug script no longer needed
    "darf_enhanced.py",        # Enhanced version now consolidated
]

BACKUP_FILES = [
    # Identify any .bak files or files with backup_ prefix
    "darf_api_server.py.bak",
]

CLEANUP_SCRIPTS = [
    # From the main_directory_cleanup_plan.md
    "backup_scripts_20250324_162206/cleanup_darf_duplicates.py",
    "backup_scripts_20250324_162206/cleanup_darf_repo.py",
    "backup_scripts_20250324_162206/cleanup_targeted.py",
    "backup_scripts_20250324_162206/darf_clean.py",
    "backup_scripts_20250324_162206/darf_cleanup_analyzer.py",
    "backup_scripts_20250324_162206/darf_cleanup_implementation.py",
    "backup_scripts_20250324_162206/final_cleanup.py",
    "backup_scripts_20250324_162206/final_consolidation.py",
    "backup_scripts_20250324_162206/fix_darf_auto.py",
    "backup_scripts_20250324_162206/fix_darf.py",
    "backup_scripts_20250324_162206/fixes_and_optimizations.py",
    "backup_scripts_20250324_162206/merge_darf_components.py",
    "backup_scripts_20250324_162206/safely_remove_duplicates.py",
    "backup_scripts_20250324_162206/simple_final_consolidation.py",
]

REDUNDANT_DIRECTORIES = [
    "DARF-Framework-Prototype",  # Nested duplicate of the main repository
    "test_vault",                # Test directory no longer needed
    "test_integration_vault",    # Test directory no longer needed
]

REDUNDANT_DOC_FILES = [
    # Analysis and optimization documents now consolidated
    "API_CONNECTOR_OPTIMIZATION.md",
    "CONSENSUS_OPTIMIZATION.md",
    "DARF_CONCURRENCY_IMPROVEMENTS.md",
    "DARF_CONSOLIDATION_SUMMARY.md",
    "DARF_SYSTEM_ANALYSIS.md",
    "DARF_SYSTEM_OPTIMIZATION_SUMMARY.md",
    "EVENT_BUS_OPTIMIZATION.md",
    "KNOWLEDGE_GRAPH_OPTIMIZATION.md",
    "LLM_VECTOR_OPTIMIZATION.md",
    "RZSET_OPTIMIZATION.md",
    "VAULT_OPTIMIZATION.md",
]

# Files that should be checked but not automatically removed
FILES_TO_CHECK = [
    "temp_config.json",  # May contain temporary configurations
]

def create_backup_directory():
    """Create a backup directory with timestamp."""
    backup_dir = Path(f"backup_{timestamp}")
    backup_dir.mkdir(exist_ok=True)
    logger.info(f"Created backup directory: {backup_dir}")
    return backup_dir

def backup_file(file_path, backup_dir):
    """Backup a file to the backup directory."""
    if not Path(file_path).exists():
        logger.warning(f"File not found, skipping backup: {file_path}")
        return False

    dest_path = backup_dir / Path(file_path).name
    try:
        shutil.copy2(file_path, dest_path)
        logger.info(f"Backed up: {file_path} -> {dest_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to backup {file_path}: {e}")
        return False

def remove_file(file_path, dry_run=False):
    """Remove a file if it exists."""
    if not Path(file_path).exists():
        logger.warning(f"File not found, skipping removal: {file_path}")
        return False

    if dry_run:
        logger.info(f"[DRY RUN] Would remove: {file_path}")
        return True

    try:
        os.remove(file_path)
        logger.info(f"Removed file: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to remove {file_path}: {e}")
        return False

def remove_directory(dir_path, dry_run=False):
    """Remove a directory if it exists."""
    if not Path(dir_path).exists():
        logger.warning(f"Directory not found, skipping removal: {dir_path}")
        return False

    if dry_run:
        logger.info(f"[DRY RUN] Would remove directory: {dir_path}")
        return True

    try:
        shutil.rmtree(dir_path)
        logger.info(f"Removed directory: {dir_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to remove directory {dir_path}: {e}")
        return False

def save_cleanup_report(stats, dry_run=False):
    """Save a report of the cleanup actions."""
    report = {
        "timestamp": timestamp,
        "dry_run": dry_run,
        "files_removed": stats["files_removed"],
        "directories_removed": stats["directories_removed"],
        "files_backed_up": stats["files_backed_up"],
        "errors": stats["errors"],
    }

    report_dir = Path("cleanup_plans")
    report_dir.mkdir(exist_ok=True)
    report_file = report_dir / f"cleanup_execution_{timestamp}.json"

    try:
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved cleanup report to: {report_file}")
    except Exception as e:
        logger.error(f"Failed to save cleanup report: {e}")

def main():
    """Main cleanup function."""
    parser = argparse.ArgumentParser(description="DARF Framework Directory Cleanup Implementation")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without making changes")
    args = parser.parse_args()

    if args.dry_run:
        logger.info("Performing DRY RUN - no files will be modified or removed")

    # Stats dictionary to track actions
    stats = {
        "files_removed": [],
        "directories_removed": [],
        "files_backed_up": [],
        "errors": [],
    }

    # Create backup directory
    backup_dir = create_backup_directory()

    # Process run scripts
    logger.info("Processing redundant run scripts...")
    for script in REDUNDANT_RUN_SCRIPTS:
        if backup_file(script, backup_dir):
            stats["files_backed_up"].append(script)
            if remove_file(script, args.dry_run):
                stats["files_removed"].append(script)

    # Process redundant script files
    logger.info("Processing redundant script files...")
    for script in REDUNDANT_SCRIPT_FILES:
        if backup_file(script, backup_dir):
            stats["files_backed_up"].append(script)
            if remove_file(script, args.dry_run):
                stats["files_removed"].append(script)

    # Process backup files
    logger.info("Processing backup files...")
    for file in BACKUP_FILES:
        if Path(file).exists() and remove_file(file, args.dry_run):
            stats["files_removed"].append(file)

    # Process cleanup scripts
    logger.info("Processing old cleanup scripts...")
    for script in CLEANUP_SCRIPTS:
        if backup_file(script, backup_dir):
            stats["files_backed_up"].append(script)
            if remove_file(script, args.dry_run):
                stats["files_removed"].append(script)

    # Process redundant directories
    logger.info("Processing redundant directories...")
    for directory in REDUNDANT_DIRECTORIES:
        if remove_directory(directory, args.dry_run):
            stats["directories_removed"].append(directory)

    # Process redundant documentation files
    logger.info("Processing redundant documentation files...")
    for doc_file in REDUNDANT_DOC_FILES:
        if backup_file(doc_file, backup_dir):
            stats["files_backed_up"].append(doc_file)
            if remove_file(doc_file, args.dry_run):
                stats["files_removed"].append(doc_file)

    # Save cleanup report
    save_cleanup_report(stats, args.dry_run)

    # Summary
    logger.info("=" * 50)
    logger.info("Cleanup Summary:")
    logger.info(f"  Files backed up: {len(stats['files_backed_up'])}")
    logger.info(f"  Files removed: {len(stats['files_removed'])}")
    logger.info(f"  Directories removed: {len(stats['directories_removed'])}")
    logger.info(f"  Errors encountered: {len(stats['errors'])}")
    logger.info("=" * 50)

    if args.dry_run:
        logger.info("This was a DRY RUN - no files were actually removed.")
        logger.info("Run without --dry-run to perform the actual cleanup.")

    if stats["errors"]:
        logger.warning("Some errors were encountered during cleanup.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
