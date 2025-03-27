#!/usr/bin/env python3
"""
Enhanced DARF Framework Directory Cleanup

This script extends the existing cleanup implementation to provide a more comprehensive
cleanup process that identifies additional redundant files and provides a safer
removal procedure with improved reporting.

Usage:
    python enhanced_darf_cleanup.py [--dry-run] [--analysis-only] [--aggressive]

Options:
    --dry-run         Simulate cleanup without making changes
    --analysis-only   Only perform analysis and generate report without removing files
    --aggressive      Include additional files in cleanup that might be redundant
"""

import os
import sys
import shutil
import argparse
import json
import datetime
import logging
import re
from pathlib import Path
import hashlib

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"enhanced_cleanup_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file),
    ],
)
logger = logging.getLogger("DARF.EnhancedCleanup")

# Known essential files (should never be removed)
ESSENTIAL_FILES = [
    "run_darf_secured.py",        # Main launcher for secured system
    "run_darf_all_in_one.py",     # Consolidated all-in-one launcher
    "run_darf_fixed.py",          # Fixed version launcher
    "run_darf_visualizer.py",     # Visualizer launcher
    "run_darf_launcher.py",       # Generic launcher
    "advanced_cache.py",          # Core caching functionality
    "darf_config.json",           # Main configuration
    "docker-compose.yml",         # Docker configuration
    "docker-compose.yml.secure",  # Secured Docker configuration
    "Dockerfile.darf",            # Docker build file
    "Dockerfile.darf.secure",     # Secured Docker build file
    "requirements.txt",           # Core dependencies
    "requirements_consolidated.txt",  # Consolidated dependencies
    "implement_main_dir_cleanup.py",  # Existing cleanup script
    "README.md",                  # Main documentation
    "README_CONSOLIDATED.md",     # Consolidated documentation
    "README_FIXED.md",            # Fixed version documentation
    "README-docker.md",           # Docker documentation
    "README-LLM-MANAGER.md",      # LLM Manager documentation
    "darf_api_server.py",         # API server
    "DARF.py",                    # Core DARF module
    "darf_core.py",               # Core DARF functionality
    "darf_dashboard_app.py",      # Dashboard application
    "darf_consolidated_dashboard.py",  # Consolidated dashboard
    "config.py",                  # Configuration module
    "auth.py",                    # Authentication module
    "async_tasks.py",             # Asynchronous tasks module
    "llm_registry.py",            # LLM registry module
    "error_handlers.py",          # Error handling module
    "users.json",                 # User information
    "api_tokens.json",            # API tokens
]

# Lists from the original cleanup script
REDUNDANT_RUN_SCRIPTS = [
    "run_darf_datasets.py",            # Duplicate identified in cleanup plan
    "run_darf_api.py",                 # Functionality covered in run_darf_secured.py
    "run_darf_dashboard_only.py",      # Limited functionality covered by other scripts
    "run_darf_frontend.py",            # Functionality covered in run_darf_secured.py
    "run_darf_minimal.py",             # Basic functionality covered by other scripts
    "run_darf_simple.py",              # Simple version superseded by more complete versions
    "run_darf_enhanced.py",            # Enhanced functionality now in consolidated versions
    "run_darf_prometheus_grafana.py",  # Functionality now in secured version
    "run_darf_integrated.py",          # Integrated functionality now in consolidated versions
    "run_darf_integrated_lifecycle.py",  # Functionality now in consolidated versions
    "run_darf_consolidated.py",        # Now using run_darf_all_in_one.py
]

REDUNDANT_SCRIPT_FILES = [
    "darf_api_server.py.bak",  # Backup file no longer needed
    "debug_darf_system.py",    # Debug script no longer needed
    "darf_enhanced.py",        # Enhanced version now consolidated
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

# Additional redundant files to consider with aggressive cleanup
AGGRESSIVE_CLEANUP_FILES = [
    # Test files (should generally be kept, but might be redundant if tests are no longer run)
    "test_consensus_protocol_selector_optimized.py",  # Test file that might be integrated elsewhere
    "test_knowledge_graph_optimized.py",              # Test file that might be integrated elsewhere
    "test_rubiks_vault_optimized.py",                 # Test file that might be integrated elsewhere
    "check_flask.py",                                 # Flask check utility, might be redundant
    "darf_fixer.py",                                  # Fixer script, might be redundant if fixes are applied
    "verify_darf_setup.py",                           # Setup verification, might be redundant if system is stable
]

# Files that should be checked but not automatically removed
FILES_TO_CHECK = [
    "temp_config.json",  # May contain temporary configurations
]

def file_hash(file_path):
    """Calculate SHA-256 hash of a file."""
    if not os.path.isfile(file_path):
        return None
    
    try:
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash for {file_path}: {e}")
        return None

def find_duplicate_files(directory):
    """Find duplicate files by content."""
    hash_dict = {}
    duplicates = []
    
    logger.info(f"Scanning for duplicate files in {directory}...")
    for root, _, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            
            # Skip certain directories
            if any(excluded in file_path for excluded in [
                ".git", "node_modules", "__pycache__", 
                "backup_", "logs", "vault_storage"
            ]):
                continue
                
            # Skip large files (> 10MB)
            if os.path.getsize(file_path) > 10_000_000:
                continue
                
            file_h = file_hash(file_path)
            if file_h:
                if file_h in hash_dict:
                    if os.path.basename(file_path) not in [os.path.basename(h) for h in hash_dict[file_h]]:
                        duplicates.append((file_path, hash_dict[file_h][0]))
                    hash_dict[file_h].append(file_path)
                else:
                    hash_dict[file_h] = [file_path]
    
    return duplicates

def find_backup_files():
    """Find backup files throughout the repository."""
    backup_files = []
    
    # Patterns that indicate backup files
    backup_patterns = [
        r"\.bak$",
        r"\.backup(_\d+)?$",
        r"\.old$",
        r"\.orig$",
        r"~$",
        r"-backup",
        r"backup-"
    ]
    
    logger.info("Scanning for backup files...")
    for root, _, files in os.walk("."):
        for filename in files:
            file_path = os.path.join(root, filename)
            
            # Skip certain directories
            if any(excluded in file_path for excluded in [
                ".git", "node_modules", "__pycache__"
            ]):
                continue
            
            for pattern in backup_patterns:
                if re.search(pattern, filename):
                    backup_files.append(file_path)
                    break
    
    return backup_files

def analyze_repository():
    """Analyze the repository for redundant files."""
    analysis = {
        "timestamp": timestamp,
        "redundant_run_scripts": [],
        "redundant_script_files": [],
        "backup_files": [],
        "redundant_directories": [],
        "redundant_doc_files": [],
        "duplicate_files": [],
        "aggressive_cleanup_files": [],
        "files_to_check": []
    }
    
    # Check run scripts
    for script in REDUNDANT_RUN_SCRIPTS:
        if os.path.exists(script):
            analysis["redundant_run_scripts"].append(script)
    
    # Check redundant script files
    for script in REDUNDANT_SCRIPT_FILES:
        if os.path.exists(script):
            analysis["redundant_script_files"].append(script)
    
    # Check redundant directories
    for directory in REDUNDANT_DIRECTORIES:
        if os.path.exists(directory) and os.path.isdir(directory):
            analysis["redundant_directories"].append(directory)
    
    # Check redundant doc files
    for doc in REDUNDANT_DOC_FILES:
        if os.path.exists(doc):
            analysis["redundant_doc_files"].append(doc)
    
    # Find backup files
    backup_files = find_backup_files()
    for backup in backup_files:
        if backup not in analysis["backup_files"]:
            analysis["backup_files"].append(backup)
    
    # Find duplicate files
    duplicate_files = find_duplicate_files(".")
    for dup in duplicate_files:
        dup_info = {"file1": dup[0], "file2": dup[1]}
        
        # Skip if both files are in ESSENTIAL_FILES
        if (os.path.basename(dup[0]) in ESSENTIAL_FILES and 
            os.path.basename(dup[1]) in ESSENTIAL_FILES):
            continue
            
        analysis["duplicate_files"].append(dup_info)
    
    # Check aggressive cleanup files
    if args.aggressive:
        for file in AGGRESSIVE_CLEANUP_FILES:
            if os.path.exists(file):
                analysis["aggressive_cleanup_files"].append(file)
    
    # Check files that need manual review
    for file in FILES_TO_CHECK:
        if os.path.exists(file):
            analysis["files_to_check"].append(file)
    
    return analysis

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

def save_analysis_report(analysis):
    """Save the analysis report to a file."""
    report_dir = Path("cleanup_plans")
    report_dir.mkdir(exist_ok=True)
    
    report_file = report_dir / f"enhanced_cleanup_analysis_{timestamp}.json"
    summary_file = report_dir / f"enhanced_cleanup_summary_{timestamp}.md"
    
    try:
        # Save JSON report
        with open(report_file, "w") as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"Saved analysis report to: {report_file}")
        
        # Create summary markdown
        with open(summary_file, "w") as f:
            f.write(f"# DARF Framework Enhanced Cleanup Analysis\n\n")
            f.write(f"Analysis timestamp: {timestamp}\n\n")
            
            f.write("## Redundant Run Scripts\n\n")
            for script in analysis["redundant_run_scripts"]:
                f.write(f"- {script}\n")
            
            f.write("\n## Redundant Script Files\n\n")
            for script in analysis["redundant_script_files"]:
                f.write(f"- {script}\n")
            
            f.write("\n## Backup Files\n\n")
            for backup in analysis["backup_files"]:
                f.write(f"- {backup}\n")
            
            f.write("\n## Redundant Directories\n\n")
            for directory in analysis["redundant_directories"]:
                f.write(f"- {directory}\n")
            
            f.write("\n## Redundant Documentation Files\n\n")
            for doc in analysis["redundant_doc_files"]:
                f.write(f"- {doc}\n")
            
            f.write("\n## Duplicate Files\n\n")
            for dup in analysis["duplicate_files"]:
                f.write(f"- {dup['file1']} (duplicate of {dup['file2']})\n")
            
            if args.aggressive:
                f.write("\n## Additional Files (Aggressive Cleanup)\n\n")
                for file in analysis["aggressive_cleanup_files"]:
                    f.write(f"- {file}\n")
            
            f.write("\n## Files To Check Manually\n\n")
            for file in analysis["files_to_check"]:
                f.write(f"- {file}\n")
            
            total_items = (len(analysis["redundant_run_scripts"]) +
                          len(analysis["redundant_script_files"]) +
                          len(analysis["backup_files"]) +
                          len(analysis["redundant_directories"]) +
                          len(analysis["redundant_doc_files"]) +
                          len(analysis["duplicate_files"]) +
                          len(analysis["aggressive_cleanup_files"]))
            
            f.write(f"\n\n## Summary\n\n")
            f.write(f"Total items identified for cleanup: {total_items}\n")
            f.write(f"Items requiring manual review: {len(analysis['files_to_check'])}\n")
            
        logger.info(f"Saved summary report to: {summary_file}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to save analysis report: {e}")
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
    report_file = report_dir / f"enhanced_cleanup_execution_{timestamp}.json"

    try:
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved cleanup report to: {report_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to save cleanup report: {e}")
        return False

def perform_cleanup(analysis, dry_run=False):
    """Perform the cleanup based on the analysis."""
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
    for script in analysis["redundant_run_scripts"]:
        if backup_file(script, backup_dir):
            stats["files_backed_up"].append(script)
            if remove_file(script, dry_run):
                stats["files_removed"].append(script)

    # Process redundant script files
    logger.info("Processing redundant script files...")
    for script in analysis["redundant_script_files"]:
        if backup_file(script, backup_dir):
            stats["files_backed_up"].append(script)
            if remove_file(script, dry_run):
                stats["files_removed"].append(script)

    # Process backup files
    logger.info("Processing backup files...")
    for file in analysis["backup_files"]:
        # Skip backup files in the current backup directory
        if str(backup_dir) in file:
            continue
            
        # Backup the backup file (for safety)
        if backup_file(file, backup_dir):
            stats["files_backed_up"].append(file)
            if remove_file(file, dry_run):
                stats["files_removed"].append(file)

    # Process redundant directories
    logger.info("Processing redundant directories...")
    for directory in analysis["redundant_directories"]:
        if remove_directory(directory, dry_run):
            stats["directories_removed"].append(directory)

    # Process redundant documentation files
    logger.info("Processing redundant documentation files...")
    for doc_file in analysis["redundant_doc_files"]:
        if backup_file(doc_file, backup_dir):
            stats["files_backed_up"].append(doc_file)
            if remove_file(doc_file, dry_run):
                stats["files_removed"].append(doc_file)

    # Process duplicate files - only if not in ESSENTIAL_FILES
    logger.info("Processing duplicate files...")
    for dup in analysis["duplicate_files"]:
        file1 = dup["file1"]
        file2 = dup["file2"]
        
        # Determine which file to keep (prefer files in ESSENTIAL_FILES)
        file_to_remove = None
        if os.path.basename(file1) in ESSENTIAL_FILES:
            file_to_remove = file2
        elif os.path.basename(file2) in ESSENTIAL_FILES:
            file_to_remove = file1
        else:
            # If neither is essential, prefer shorter paths and non-backup files
            if ("backup" in file1.lower() or "bak" in file1.lower() or 
                "old" in file1.lower() or "~" in file1):
                file_to_remove = file1
            elif ("backup" in file2.lower() or "bak" in file2.lower() or 
                  "old" in file2.lower() or "~" in file2):
                file_to_remove = file2
            else:
                # Otherwise, keep the file with the shorter path
                file_to_remove = file1 if len(file1) > len(file2) else file2
        
        if file_to_remove:
            if backup_file(file_to_remove, backup_dir):
                stats["files_backed_up"].append(file_to_remove)
                if remove_file(file_to_remove, dry_run):
                    stats["files_removed"].append(file_to_remove)

    # Process aggressive cleanup files
    if args.aggressive:
        logger.info("Processing aggressive cleanup files...")
        for file in analysis["aggressive_cleanup_files"]:
            if backup_file(file, backup_dir):
                stats["files_backed_up"].append(file)
                if remove_file(file, dry_run):
                    stats["files_removed"].append(file)

    # Save cleanup report
    save_cleanup_report(stats, dry_run)

    # Summary
    logger.info("=" * 50)
    logger.info("Cleanup Summary:")
    logger.info(f"  Files backed up: {len(stats['files_backed_up'])}")
    logger.info(f"  Files removed: {len(stats['files_removed'])}")
    logger.info(f"  Directories removed: {len(stats['directories_removed'])}")
    logger.info(f"  Errors encountered: {len(stats['errors'])}")
    logger.info("=" * 50)

    if dry_run:
        logger.info("This was a DRY RUN - no files were actually removed.")
        logger.info("Run without --dry-run to perform the actual cleanup.")

    return stats

def main():
    """Main function."""
    if args.dry_run:
        logger.info("Performing DRY RUN - no files will be modified or removed")
        
    # First, perform repository analysis
    logger.info("Analyzing repository for redundant files...")
    analysis = analyze_repository()
    
    # Save analysis report
    save_analysis_report(analysis)
    
    # If analysis only, stop here
    if args.analysis_only:
        logger.info("Analysis completed. Check the cleanup_plans directory for the report.")
        return 0
    
    # Otherwise, perform cleanup
    logger.info("Performing cleanup based on analysis...")
    stats = perform_cleanup(analysis, args.dry_run)
    
    if stats["errors"]:
        logger.warning("Some errors were encountered during cleanup.")
        return 1
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced DARF Framework Directory Cleanup")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without making changes")
    parser.add_argument("--analysis-only", action="store_true", help="Only perform analysis, no cleanup")
    parser.add_argument("--aggressive", action="store_true", help="Include additional files in cleanup")
    args = parser.parse_args()
    
    sys.exit(main())
