"""
DARF Multi-Dimensional Debug Compatibility Layer

This module provides compatibility layers for resolving interface mismatches 
between different DARF components, with multi-dimensional debugging capabilities:

1. API Compatibility - Mapping between different method signatures
2. Data Structure Tracing - Runtime inspection of data transformations
3. Error Dimensional Analysis - Correlation of errors across components
4. Execution Flow Visualization - Tracing call paths through the system
5. Thread/Async Boundary Inspection - Detecting concurrency issues
"""

import os
import sys
import json
import time
import inspect
import logging
import threading
import asyncio
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from datetime import datetime

# Configure logging
logger = logging.getLogger("DARF.Debug")

class DimensionalDebugger:
    """Multi-dimensional debug capability provider."""
    
    def __init__(self, component_name: str, context: Dict[str, Any] = None):
        self.component_name = component_name
        self.context = context or {}
        self.call_stack = []
        self.transaction_id = None
        self.dimensions = {
            "api": {},
            "data": {},
            "error": {},
            "flow": {},
            "concurrency": {}
        }
        
        # Thread-local storage for tracking concurrent operations
        self.local = threading.local()
        
        logger.info(f"Dimensional Debugger initialized for {component_name}")
    
    def trace_call(self, method_name: str, args: tuple = None, kwargs: dict = None):
        """Trace a method call with dimensional debugging."""
        # Generate a transaction ID if not present
        if not hasattr(self.local, 'transaction_id'):
            self.local.transaction_id = f"{time.time()}-{threading.get_ident()}"
            self.local.call_depth = 0
        else:
            self.local.call_depth += 1
        
        # Format args and kwargs for logging
        args_str = ", ".join([str(arg)[:100] for arg in (args or [])])
        kwargs_str = ", ".join([f"{k}={str(v)[:100]}" for k, v in (kwargs or {}).items()])
        params = f"{args_str}{', ' if args_str and kwargs_str else ''}{kwargs_str}"
        
        # Log call with proper indentation
        indent = "  " * self.local.call_depth
        caller_frame = inspect.currentframe().f_back
        caller_info = f"{os.path.basename(caller_frame.f_code.co_filename)}:{caller_frame.f_lineno}"
        logger.debug(f"{indent}[{self.local.transaction_id}] {self.component_name}.{method_name}({params}) from {caller_info}")
        
        # Store API dimension data
        self.dimensions["api"][method_name] = {
            "timestamp": datetime.now().isoformat(),
            "args": args,
            "kwargs": kwargs,
            "caller": caller_info,
            "transaction_id": self.local.transaction_id,
            "thread_id": threading.get_ident(),
            "is_async": asyncio.iscoroutinefunction(caller_frame.f_code)
        }
        
        # Store flow dimension data
        if "calls" not in self.dimensions["flow"]:
            self.dimensions["flow"]["calls"] = []
        
        self.dimensions["flow"]["calls"].append({
            "method": method_name,
            "timestamp": datetime.now().isoformat(),
            "depth": self.local.call_depth,
            "transaction_id": self.local.transaction_id
        })
        
        return self.local.transaction_id, self.local.call_depth
    
    def trace_return(self, method_name: str, result: Any, call_info: tuple = None):
        """Trace a method return with dimensional debugging."""
        transaction_id, call_depth = call_info or (self.local.transaction_id, self.local.call_depth)
        
        # Log return with proper indentation
        indent = "  " * call_depth
        result_str = str(result)[:100]
        logger.debug(f"{indent}[{transaction_id}] {self.component_name}.{method_name} => {result_str}")
        
        # Store data dimension info
        self.dimensions["data"][method_name] = {
            "timestamp": datetime.now().isoformat(),
            "result": result,
            "transaction_id": transaction_id
        }
        
        # Decrement call depth
        if hasattr(self.local, 'call_depth'):
            self.local.call_depth -= 1
        
        return result
    
    def trace_error(self, method_name: str, error: Exception, call_info: tuple = None):
        """Trace an error with dimensional debugging."""
        transaction_id, call_depth = call_info or (self.local.transaction_id, self.local.call_depth)
        
        # Log error with proper indentation
        indent = "  " * call_depth
        logger.error(f"{indent}[{transaction_id}] {self.component_name}.{method_name} => ERROR: {str(error)}")
        
        # Store error dimension info
        if method_name not in self.dimensions["error"]:
            self.dimensions["error"][method_name] = []
        
        self.dimensions["error"][method_name].append({
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "transaction_id": transaction_id,
            "traceback": getattr(error, "__traceback__", None)
        })
        
        # Decrement call depth
        if hasattr(self.local, 'call_depth'):
            self.local.call_depth -= 1
        
        return error
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get all collected debug information."""
        return {
            "component": self.component_name,
            "dimensions": self.dimensions,
            "context": self.context,
            "timestamp": datetime.now().isoformat()
        }
    
    def save_debug_info(self, filepath: str = None) -> str:
        """Save debug information to a file."""
        if filepath is None:
            os.makedirs("logs", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"logs/darf_debug_{self.component_name}_{timestamp}.json"
        
        with open(filepath, "w") as f:
            json.dump(self.get_debug_info(), f, indent=2, default=str)
        
        logger.info(f"Debug information saved to {filepath}")
        return filepath

class KnowledgeGraphCompat:
    """
    Compatibility layer for the KnowledgeGraph class.
    
    This adds backward compatibility for older interfaces while enabling
    multi-dimensional debugging of the knowledge graph operations.
    """
    
    def __init__(self, kg_instance):
        """
        Initialize the compatibility layer.
        
        Args:
            kg_instance: Knowledge Graph instance to wrap
        """
        self.kg = kg_instance
        self.debugger = DimensionalDebugger("KnowledgeGraph")
        logger.info("KnowledgeGraph compatibility layer initialized")
    
    def add_fact(self, fact: Dict[str, Any]) -> bool:
        """
        Add a fact to the knowledge graph (compatibility method).
        
        Maps to add_node in the enhanced implementation.
        
        Args:
            fact: Fact data (dict with subject, predicate, object)
            
        Returns:
            Success status
        """
        call_info = self.debugger.trace_call("add_fact", (fact,))
        
        try:
            # Convert fact format to node format
            node_data = self._convert_fact_to_node(fact)
            
            # Generate node ID from subject if possible
            node_id = fact.get("subject", None)
            
            # Create the async task for add_node
            loop = asyncio.get_event_loop()
            if asyncio.iscoroutinefunction(self.kg.add_node):
                task = loop.create_task(self.kg.add_node(node_data, node_id))
                node_id = loop.run_until_complete(task)
            else:
                node_id = self.kg.add_node(node_data, node_id)
            
            # If subject is different from assigned node_id, create a relationship
            if node_id != fact.get("subject"):
                # We'd set up relationships here if needed
                pass
            
            logger.info(f"Added fact: {fact}")
            return self.debugger.trace_return("add_fact", True, call_info)
        except Exception as e:
            logger.error(f"Error adding fact: {e}")
            self.debugger.trace_error("add_fact", e, call_info)
            return False
    
    def _convert_fact_to_node(self, fact: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a fact dictionary to a node dictionary."""
        # Extract standard fact fields
        subject = fact.get("subject")
        predicate = fact.get("predicate")
        obj = fact.get("object")
        confidence = fact.get("confidence", 1.0)
        
        # Create node data
        node_data = {
            "type": "fact",
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "confidence": confidence
        }
        
        # Add any additional fields
        for key, value in fact.items():
            if key not in ["subject", "predicate", "object", "confidence"]:
                node_data[key] = value
        
        return node_data
    
    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the knowledge graph (compatibility method).
        
        Returns:
            Summary data
        """
        call_info = self.debugger.trace_call("summary")
        
        try:
            # Get facts count
            loop = asyncio.get_event_loop()
            if asyncio.iscoroutinefunction(self.kg.get_fact_count):
                task = loop.create_task(self.kg.get_fact_count())
                fact_count = loop.run_until_complete(task)
            else:
                fact_count = self.kg.get_fact_count() if hasattr(self.kg, "get_fact_count") else len(self.kg.get_facts())
            
            # Get relationship types
            rel_types = self.kg.get_relationship_types()
            
            # Get node metrics if available
            metrics = {}
            if hasattr(self.kg, "get_metrics"):
                if asyncio.iscoroutinefunction(self.kg.get_metrics):
                    task = loop.create_task(self.kg.get_metrics())
                    metrics = loop.run_until_complete(task)
                else:
                    metrics = self.kg.get_metrics()
            
            # Build summary
            summary = {
                "total_facts": fact_count,
                "relationship_types": rel_types,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add metrics if available
            if metrics:
                for key, value in metrics.items():
                    if key not in summary:
                        summary[key] = value
            
            return self.debugger.trace_return("summary", summary, call_info)
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            self.debugger.trace_error("summary", e, call_info)
            return {"total_facts": 0, "error": str(e)}
    
    def __getattr__(self, name):
        """
        Pass through any other attributes to the underlying KnowledgeGraph instance.
        
        This allows the compatibility layer to be used as a drop-in replacement.
        """
        return getattr(self.kg, name)
    
    def save_debug_info(self):
        """Save debug information to a file."""
        return self.debugger.save_debug_info()

def patch_kg_instance(kg_instance):
    """
    Patch a KnowledgeGraph instance with compatibility methods.
    
    Args:
        kg_instance: Knowledge Graph instance to patch
        
    Returns:
        Patched KnowledgeGraph instance
    """
    return KnowledgeGraphCompat(kg_instance)

# Instrumentation for multi-dimensional debugging of the DARF system
class DARFSystemDebugger:
    """System-wide debugger for multi-dimensional analysis."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the system debugger."""
        self.component_debuggers = {}
        self.system_events = []
        self.error_correlations = {}
        self.performance_metrics = {}
        
        # Set up logging
        self.setup_logging()
        
        # Track startup time
        self.startup_time = datetime.now()
        logger.info("DARF System Debugger initialized")
    
    def setup_logging(self):
        """Set up logging for the system debugger."""
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        
        # Set up file handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(f"logs/darf_debug_{timestamp}.log")
        file_handler.setLevel(logging.DEBUG)
        
        # Set up formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)
    
    def register_component(self, component_name: str, instance: Any):
        """Register a component for debugging."""
        debugger = DimensionalDebugger(component_name, {"instance_type": type(instance).__name__})
        self.component_debuggers[component_name] = debugger
        logger.info(f"Registered component {component_name} for debugging")
        return debugger
    
    def log_system_event(self, event_type: str, data: Dict[str, Any] = None):
        """Log a system-wide event."""
        event = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data or {}
        }
        self.system_events.append(event)
        logger.info(f"System event: {event_type}")
    
    def correlate_errors(self):
        """Correlate errors across components."""
        all_errors = {}
        
        # Collect all errors
        for component_name, debugger in self.component_debuggers.items():
            for method_name, errors in debugger.dimensions["error"].items():
                for error in errors:
                    error_key = f"{error['error_type']}:{error['error_message']}"
                    if error_key not in all_errors:
                        all_errors[error_key] = []
                    all_errors[error_key].append({
                        "component": component_name,
                        "method": method_name,
                        "timestamp": error["timestamp"],
                        "transaction_id": error["transaction_id"]
                    })
        
        # Sort by frequency
        self.error_correlations = {
            error_key: instances
            for error_key, instances in sorted(
                all_errors.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )
        }
        
        return self.error_correlations
    
    def save_debug_report(self) -> str:
        """Save a comprehensive debug report."""
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"logs/darf_debug_report_{timestamp}.json"
        
        # Correlate errors
        self.correlate_errors()
        
        # Build report
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_uptime": (datetime.now() - self.startup_time).total_seconds(),
            "components": {
                name: debugger.get_debug_info()
                for name, debugger in self.component_debuggers.items()
            },
            "system_events": self.system_events,
            "error_correlations": self.error_correlations,
            "performance_metrics": self.performance_metrics
        }
        
        # Save report
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Debug report saved to {filepath}")
        return filepath

# Initialize system debugger
system_debugger = DARFSystemDebugger.get_instance()
