# Main Directory Cleanup Plan

After analyzing the main directory contents, I've identified several opportunities for consolidation and cleanup.

## Scripts to Remove (Cleanup-related scripts that have already done their job)

1. **cleanup_darf_duplicates.py** - Used for identifying duplicate implementations
2. **cleanup_darf_repo.py** - General repository cleanup script
3. **darf_clean.py** - Simplified cleanup script
4. **darf_cleanup_analyzer.py** - Analysis script for cleanup
5. **darf_cleanup_implementation.py** - Cleanup implementation script
6. **final_cleanup.py** - Script for handling stubborn directories
7. **cleanup_targeted.py** - Targeted cleanup script (we created this)
8. **final_consolidation.py** - Consolidation script (we created this)
9. **simple_final_consolidation.py** - Simplified consolidation script (we created this)
10. **safely_remove_duplicates.py** - Script for safely removing duplicates

## Scripts to Consolidate (Fix and Optimization scripts)

These scripts have similar functionality and can be consolidated into a single "darf_fixer.py" script:

1. **fix_darf.py** - For diagnosing and fixing issues
2. **fix_darf_auto.py** - Automatic detection and fixing of issues
3. **fixes_and_optimizations.py** - For fixing bottlenecks and optimizations
4. **merge_darf_components.py** - For merging components

## Scripts to Keep

1. **Core Framework Files**:
   - **DARF.py** - Main framework file
   - **darf_core.py** - Core functionality
   - **darf_cli.py** - CLI interface

2. **Run Scripts** (already consolidated):
   - **run_darf_all_in_one.py** - For running all components
   - **run_darf_datasets.py** - For dataset operations
   - **run_darf_fixed.py** - Fixed implementation
   - **run_darf_launcher.py** - Launcher script
   - **run_darf_visualizer.py** - Visualization tools

3. **Verification Scripts**:
   - **verify_darf_setup.py** - For verifying setup
   - **check_flask.py** - For checking Flask dependencies

## Configuration Files to Keep

1. **darf_config.json** - Main configuration file
2. **default_config.json** - Default configuration

## Analysis Reports to Remove

1. **duplicate_files_report.json** - No longer needed after cleanup
2. **run_scripts_analysis.json** - No longer needed after consolidation

## Consolidation Strategy

1. Create a new script called `darf_fixer.py` that combines functionality from:
   - fix_darf.py
   - fix_darf_auto.py
   - fixes_and_optimizations.py
   - merge_darf_components.py

2. Remove all cleanup-related scripts as they've done their job.

3. Remove the analysis reports.

This cleanup will significantly reduce the number of files in the main directory while preserving all important functionality.
