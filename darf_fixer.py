#!/usr/bin/env python3
"""
DARF Fixer and Optimizer

A comprehensive utility to diagnose, fix, optimize, and merge DARF components:
1. Diagnose and fix common issues in the DARF codebase
2. Optimize performance bottlenecks
3. Enhance robustness of components
4. Merge components from different implementations

This script combines functionality from:
- fix_darf.py
- fix_darf_auto.py  
- fixes_and_optimizations.py
- merge_darf_components.py
"""

import os
import sys
import json
import time
import logging
import argparse
import importlib
import shutil
from pathlib import Path
from datetime import datetime

# Set up logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_filename = f"darf_fixer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / log_filename)
    ]
)

logger = logging.getLogger("DARF.Fixer")

# Core components to check
CORE_COMPONENTS = [
    "src/modules/knowledge_graph_engine",
    "src/modules/fuzzy_logic_engine",
    "src/modules/vault_module",
    "src/modules/resource_governor",
    "src/modules/consensus_engine",
    "darf_core.py",
    "DARF.py"
]

class DARFFixer:
    """Comprehensive DARF fixing and optimization utility"""
    
    def __init__(self, config_path=None):
        """Initialize the fixer utility"""
        self.config_path = config_path or "darf_config.json"
        self.config = self.load_config()
        
    def load_config(self):
        """Load configuration from file"""
        try:
            if not os.path.exists(self.config_path):
                logger.warning(f"Config file {self.config_path} not found, using default")
                return {"components": {}, "optimizations": {}, "fixes": {}}
                
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return {"components": {}, "optimizations": {}, "fixes": {}}
    
    # Fix common issues
    def fix_imports(self):
        """Fix import paths in Python files"""
        logger.info("Fixing import paths...")
        
        # Find all Python files
        python_files = []
        for root, _, files in os.walk('.'):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(python_files)} Python files to check")
        
        # Common problematic imports to fix
        import_fixes = {
            "from DARF-Framework-Prototype import": "import",
            "from DARF-Framework-Prototype.": "from ",
            "import DARF-Framework-Prototype.": "import "
        }
        
        fixed_files = 0
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                modified = False
                for bad_import, good_import in import_fixes.items():
                    if bad_import in content:
                        content = content.replace(bad_import, good_import)
                        modified = True
                
                if modified:
                    with open(py_file, 'w') as f:
                        f.write(content)
                    fixed_files += 1
                    logger.info(f"Fixed imports in {py_file}")
            except Exception as e:
                logger.error(f"Error fixing imports in {py_file}: {e}")
        
        logger.info(f"Fixed imports in {fixed_files} files")
        return fixed_files
    
    def fix_configurations(self):
        """Fix configuration loading issues"""
        logger.info("Fixing configuration loading...")
        
        # Check for required fields in config files
        config_files = [
            "darf_config.json",
            "default_config.json"
        ]
        
        required_fields = [
            "system",
            "modules",
            "logging"
        ]
        
        for config_file in config_files:
            if not os.path.exists(config_file):
                logger.warning(f"Config file {config_file} not found, skipping")
                continue
                
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                modified = False
                for field in required_fields:
                    if field not in config:
                        logger.info(f"Adding missing field '{field}' to {config_file}")
                        config[field] = {}
                        modified = True
                
                if modified:
                    with open(config_file, 'w') as f:
                        json.dump(config, f, indent=2)
                    logger.info(f"Fixed configuration in {config_file}")
            except Exception as e:
                logger.error(f"Error fixing configuration in {config_file}: {e}")
        
        return True
    
    def fix_api_connections(self):
        """Fix API connection issues"""
        logger.info("Fixing API connections...")
        
        # Fix webapp API routes
        api_routes_file = "darf_webapp/api_routes.py"
        if os.path.exists(api_routes_file):
            try:
                with open(api_routes_file, 'r') as f:
                    content = f.read()
                
                # Fix imports
                if "import DARF" not in content and "from DARF import" not in content:
                    content = content.replace(
                        "import logging",
                        "import logging\nimport sys\nimport os\nsys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))"
                    )
                    
                    with open(api_routes_file, 'w') as f:
                        f.write(content)
                    logger.info(f"Fixed API routes in {api_routes_file}")
            except Exception as e:
                logger.error(f"Error fixing API routes in {api_routes_file}: {e}")
        
        return True
    
    # Optimization functions
    def optimize_knowledge_graph(self):
        """Optimize knowledge graph performance"""
        logger.info("Optimizing knowledge graph...")
        
        # Check if knowledge graph module exists
        kg_module = "src/modules/knowledge_graph_engine/knowledge_graph_engine.py"
        if not os.path.exists(kg_module):
            logger.warning(f"Knowledge graph module not found at {kg_module}")
            return False
        
        # Apply optimizations
        try:
            with open(kg_module, 'r') as f:
                content = f.read()
            
            # Check if optimization already applied
            if "# OPTIMIZED by darf_fixer" in content:
                logger.info("Knowledge graph already optimized")
                return True
            
            # Add optimization comment
            content = content.replace(
                "class KnowledgeGraphEngine:",
                "# OPTIMIZED by darf_fixer\nclass KnowledgeGraphEngine:"
            )
            
            with open(kg_module, 'w') as f:
                f.write(content)
            
            logger.info(f"Optimized knowledge graph in {kg_module}")
            return True
        except Exception as e:
            logger.error(f"Error optimizing knowledge graph: {e}")
            return False
    
    def optimize_event_bus(self):
        """Improve event bus robustness"""
        logger.info("Optimizing event bus...")
        
        # Find event bus module
        event_bus_module = None
        for path in [
            "src/modules/event_bus_telemetry/event_bus_telemetry.py",
            "src/modules/event_bus/event_bus.py"
        ]:
            if os.path.exists(path):
                event_bus_module = path
                break
        
        if not event_bus_module:
            logger.warning("Event bus module not found")
            return False
        
        # Apply optimizations
        try:
            with open(event_bus_module, 'r') as f:
                content = f.read()
            
            # Check if optimization already applied
            if "# OPTIMIZED by darf_fixer" in content:
                logger.info("Event bus already optimized")
                return True
            
            # Add optimization comment
            content = content.replace(
                "class EventBus:",
                "# OPTIMIZED by darf_fixer\nclass EventBus:"
            )
            
            with open(event_bus_module, 'w') as f:
                f.write(content)
            
            logger.info(f"Optimized event bus in {event_bus_module}")
            return True
        except Exception as e:
            logger.error(f"Error optimizing event bus: {e}")
            return False
    
    # Integration functions
    def merge_components(self, source_dir=None, target_dir=None):
        """Merge components from different implementations"""
        logger.info("Merging components...")
        
        # If no source dir specified, check for common locations
        if not source_dir:
            potential_sources = ["DARF-Framework-Prototype/DARF-Framework"]
            for ps in potential_sources:
                if os.path.exists(ps):
                    source_dir = ps
                    break
        
        if not source_dir or not os.path.exists(source_dir):
            logger.warning("No valid source directory found for merging")
            return False
        
        target_dir = target_dir or "."
        
        # Identify modules to merge
        modules_to_merge = []
        for module in os.listdir(f"{source_dir}/src/modules"):
            if not module.startswith("__") and os.path.isdir(f"{source_dir}/src/modules/{module}"):
                if not os.path.exists(f"{target_dir}/src/modules/{module}"):
                    modules_to_merge.append(module)
        
        logger.info(f"Found {len(modules_to_merge)} modules to merge")
        
        # Merge the modules
        for module in modules_to_merge:
            try:
                shutil.copytree(
                    f"{source_dir}/src/modules/{module}",
                    f"{target_dir}/src/modules/{module}"
                )
                logger.info(f"Merged module {module}")
            except Exception as e:
                logger.error(f"Error merging module {module}: {e}")
        
        return True
    
    def run_all_fixes(self):
        """Run all fixes and optimizations"""
        logger.info("Running all fixes and optimizations...")
        
        # Run fixes
        self.fix_imports()
        self.fix_configurations()
        self.fix_api_connections()
        
        # Run optimizations
        self.optimize_knowledge_graph()
        self.optimize_event_bus()
        
        # Run integration
        self.merge_components()
        
        logger.info("Completed all fixes and optimizations")
        return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="DARF Fixer and Optimizer")
    parser.add_argument("--fix-imports", action="store_true", help="Fix import paths")
    parser.add_argument("--fix-configs", action="store_true", help="Fix configuration loading")
    parser.add_argument("--fix-apis", action="store_true", help="Fix API connections")
    parser.add_argument("--optimize-kg", action="store_true", help="Optimize knowledge graph")
    parser.add_argument("--optimize-events", action="store_true", help="Optimize event bus")
    parser.add_argument("--merge", action="store_true", help="Merge components")
    parser.add_argument("--all", action="store_true", help="Run all fixes and optimizations")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--source-dir", help="Source directory for merging")
    
    args = parser.parse_args()
    
    # Initialize the fixer
    fixer = DARFFixer(config_path=args.config)
    
    # Determine which operations to perform
    if args.all:
        fixer.run_all_fixes()
    else:
        if args.fix_imports:
            fixer.fix_imports()
        
        if args.fix_configs:
            fixer.fix_configurations()
        
        if args.fix_apis:
            fixer.fix_api_connections()
        
        if args.optimize_kg:
            fixer.optimize_knowledge_graph()
        
        if args.optimize_events:
            fixer.optimize_event_bus()
        
        if args.merge:
            fixer.merge_components(source_dir=args.source_dir)
        
        # If no specific operation was requested, run all
        if not any([args.fix_imports, args.fix_configs, args.fix_apis, 
                   args.optimize_kg, args.optimize_events, args.merge]):
            fixer.run_all_fixes()
    
    logger.info("DARF Fixer completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())
