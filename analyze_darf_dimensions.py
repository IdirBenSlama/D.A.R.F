#!/usr/bin/env python3
"""
DARF Multi-Dimensional Analysis Tool

This tool analyzes DARF debug logs across multiple dimensions to identify 
interconnected issues and provide a comprehensive view of system behavior.

Usage:
  python analyze_darf_dimensions.py [--report-path PATH] [--visualize] [--dimensions DIM1,DIM2]

Dimensions:
  - api: API compatibility issues
  - data: Data transformation issues
  - error: Error patterns and correlations
  - flow: Execution flow and call paths
  - concurrency: Thread and async boundary issues
"""

import os
import sys
import json
import time
import argparse
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("DARF.Analysis")

# Check for visualization dependencies
try:
    import matplotlib.pyplot as plt
    import networkx as nx
    VISUALIZE_AVAILABLE = True
except ImportError:
    VISUALIZE_AVAILABLE = False
    logger.warning("Visualization libraries not available. Install with: pip install matplotlib networkx")

class DimensionalAnalyzer:
    """Analyzer for multi-dimensional DARF debug data."""
    
    def __init__(self, report_path: Optional[str] = None):
        """
        Initialize the analyzer.
        
        Args:
            report_path: Path to DARF debug report JSON file
        """
        self.report_path = report_path
        self.report_data = None
        self.dimension_analyses = {
            "api": {},
            "data": {},
            "error": {},
            "flow": {},
            "concurrency": {}
        }
        self.findings = []
        
        if report_path:
            self.load_report(report_path)
        else:
            # Find most recent report
            self._find_latest_report()
    
    def _find_latest_report(self):
        """Find the most recent debug report."""
        logs_dir = Path("logs")
        if not logs_dir.exists():
            logger.error("Logs directory not found")
            return
        
        # Look for debug reports
        reports = list(logs_dir.glob("darf_debug_report_*.json"))
        if not reports:
            logger.error("No debug reports found in logs directory")
            return
        
        # Sort by modification time
        reports.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        # Use most recent
        self.report_path = str(reports[0])
        logger.info(f"Using most recent report: {self.report_path}")
        self.load_report(self.report_path)
    
    def load_report(self, report_path: str):
        """Load a debug report."""
        try:
            with open(report_path, 'r') as f:
                self.report_data = json.load(f)
            logger.info(f"Loaded debug report from {report_path}")
            
            # Extract timestamp and uptime
            self.timestamp = self.report_data.get("timestamp", "Unknown")
            self.uptime = self.report_data.get("system_uptime", 0)
            
        except Exception as e:
            logger.error(f"Error loading report: {e}")
    
    def analyze_all_dimensions(self):
        """Analyze all dimensions."""
        if not self.report_data:
            logger.error("No report data available for analysis")
            return
        
        self.analyze_api_dimension()
        self.analyze_data_dimension()
        self.analyze_error_dimension()
        self.analyze_flow_dimension()
        self.analyze_concurrency_dimension()
        
        # Cross-dimensional analysis
        self._correlate_dimensions()
        
        logger.info(f"Analysis complete with {len(self.findings)} findings")
    
    def analyze_api_dimension(self):
        """Analyze API compatibility dimension."""
        api_issues = []
        
        # Process each component
        for comp_name, comp_data in self.report_data.get("components", {}).items():
            dimensions = comp_data.get("dimensions", {})
            
            # Extract API calls
            api_calls = dimensions.get("api", {})
            
            # Check for potential issues
            for method_name, call_data in api_calls.items():
                # Check for compatibility API methods
                if method_name in ["add_fact", "summary"]:
                    api_issues.append({
                        "component": comp_name,
                        "method": method_name,
                        "issue": "Compatibility method used",
                        "timestamp": call_data.get("timestamp"),
                        "caller": call_data.get("caller")
                    })
        
        # Store analysis results
        self.dimension_analyses["api"]["issues"] = api_issues
        
        # Add findings
        if api_issues:
            self.findings.append({
                "dimension": "api",
                "severity": "medium",
                "title": f"Found {len(api_issues)} API compatibility issues",
                "details": "Components are using compatibility layer methods instead of native APIs"
            })
        
        logger.info(f"API analysis complete: found {len(api_issues)} issues")
    
    def analyze_data_dimension(self):
        """Analyze data transformation dimension."""
        data_issues = []
        
        # Process each component
        for comp_name, comp_data in self.report_data.get("components", {}).items():
            dimensions = comp_data.get("dimensions", {})
            
            # Extract data transformations
            data_transforms = dimensions.get("data", {})
            
            # Check for data type transformations or serialization
            for method_name, transform_data in data_transforms.items():
                # Check for transformations between incompatible data types
                if method_name == "_convert_fact_to_node":
                    data_issues.append({
                        "component": comp_name,
                        "method": method_name,
                        "issue": "Data structure transformation",
                        "timestamp": transform_data.get("timestamp"),
                        "transformation": "fact → node"
                    })
        
        # Store analysis results
        self.dimension_analyses["data"]["issues"] = data_issues
        
        # Add findings
        if data_issues:
            self.findings.append({
                "dimension": "data",
                "severity": "low",
                "title": f"Found {len(data_issues)} data transformations",
                "details": "Data structures are being transformed between components"
            })
        
        logger.info(f"Data analysis complete: found {len(data_issues)} issues")
    
    def analyze_error_dimension(self):
        """Analyze error dimension."""
        
        # Extract error correlations
        error_correlations = self.report_data.get("error_correlations", {})
        
        # Find most common errors
        common_errors = []
        for error_key, instances in error_correlations.items():
            error_type, error_message = error_key.split(":", 1)
            common_errors.append({
                "error_type": error_type,
                "error_message": error_message,
                "count": len(instances),
                "components": list(set(i["component"] for i in instances)),
                "methods": list(set(i["method"] for i in instances)),
                "first_timestamp": min(i["timestamp"] for i in instances),
                "last_timestamp": max(i["timestamp"] for i in instances)
            })
        
        # Sort by count
        common_errors.sort(key=lambda x: x["count"], reverse=True)
        
        # Store analysis results
        self.dimension_analyses["error"]["common_errors"] = common_errors
        
        # Add findings
        if common_errors:
            for error in common_errors[:3]:  # Top 3 errors
                self.findings.append({
                    "dimension": "error",
                    "severity": "high" if error["count"] > 5 else "medium",
                    "title": f"{error['error_type']} occurred {error['count']} times",
                    "details": f"Error: {error['error_message']}\nAffected components: {', '.join(error['components'])}"
                })
        
        logger.info(f"Error analysis complete: found {len(common_errors)} error patterns")
    
    def analyze_flow_dimension(self):
        """Analyze execution flow dimension."""
        flow_issues = []
        call_paths = {}
        
        # Process each component
        for comp_name, comp_data in self.report_data.get("components", {}).items():
            dimensions = comp_data.get("dimensions", {})
            
            # Extract flow data
            flow_data = dimensions.get("flow", {})
            calls = flow_data.get("calls", [])
            
            # Group by transaction ID
            transactions = {}
            for call in calls:
                tx_id = call.get("transaction_id")
                if tx_id not in transactions:
                    transactions[tx_id] = []
                transactions[tx_id].append(call)
            
            # Analyze each transaction
            for tx_id, tx_calls in transactions.items():
                # Sort by timestamp
                tx_calls.sort(key=lambda c: c.get("timestamp", ""))
                
                # Build call path
                call_path = [c.get("method") for c in tx_calls]
                
                # Store call path
                call_paths[tx_id] = {
                    "component": comp_name,
                    "path": call_path,
                    "start": tx_calls[0].get("timestamp") if tx_calls else None,
                    "end": tx_calls[-1].get("timestamp") if tx_calls else None
                }
                
                # Check for potential issues (long chains, recursion, etc.)
                if len(call_path) > 10:
                    flow_issues.append({
                        "component": comp_name,
                        "transaction_id": tx_id,
                        "issue": "Long call chain",
                        "details": f"Call chain with {len(call_path)} methods",
                        "start": tx_calls[0].get("timestamp") if tx_calls else None
                    })
        
        # Store analysis results
        self.dimension_analyses["flow"]["issues"] = flow_issues
        self.dimension_analyses["flow"]["call_paths"] = call_paths
        
        # Add findings
        if flow_issues:
            self.findings.append({
                "dimension": "flow",
                "severity": "medium",
                "title": f"Found {len(flow_issues)} complex execution flows",
                "details": "Long call chains may indicate overly complex interactions"
            })
        
        logger.info(f"Flow analysis complete: found {len(flow_issues)} flow issues")
    
    def analyze_concurrency_dimension(self):
        """Analyze concurrency dimension."""
        concurrency_issues = []
        
        # Extract system events for initialization and startup
        system_events = self.report_data.get("system_events", [])
        
        # Find component initialization events
        init_events = [e for e in system_events if e.get("type") == "component_initialize"]
        
        # Check for potential race conditions or ordering issues
        component_order = [(e.get("data", {}).get("component"), e.get("timestamp")) 
                         for e in init_events]
        
        # Check if any components initiated before their dependencies
        # This is a simplified placeholder analysis
        if "knowledge_graph" in [c[0] for c in component_order] and "event_bus" in [c[0] for c in component_order]:
            kg_time = next((t for c, t in component_order if c == "knowledge_graph"), None)
            eb_time = next((t for c, t in component_order if c == "event_bus"), None)
            
            if kg_time and eb_time and kg_time > eb_time:
                concurrency_issues.append({
                    "issue": "Component ordering",
                    "details": "EventBus initialized before KnowledgeGraph but may have dependencies on it",
                    "components": ["KnowledgeGraph", "EventBus"]
                })
        
        # Store analysis results
        self.dimension_analyses["concurrency"]["issues"] = concurrency_issues
        
        # Add findings
        if concurrency_issues:
            self.findings.append({
                "dimension": "concurrency",
                "severity": "medium",
                "title": f"Found {len(concurrency_issues)} potential concurrency issues",
                "details": "Component initialization order may not respect dependencies"
            })
        
        logger.info(f"Concurrency analysis complete: found {len(concurrency_issues)} issues")
    
    def _correlate_dimensions(self):
        """Correlate issues across dimensions."""
        # Look for patterns where API issues lead to data issues
        api_issues = self.dimension_analyses["api"].get("issues", [])
        data_issues = self.dimension_analyses["data"].get("issues", [])
        error_issues = self.dimension_analyses["error"].get("common_errors", [])
        
        # Check for API compatibility methods that also have data transformation
        api_methods = set(issue["method"] for issue in api_issues)
        data_methods = set(issue["method"] for issue in data_issues)
        
        overlap = api_methods.intersection(data_methods)
        if overlap:
            self.findings.append({
                "dimension": "cross",
                "severity": "high",
                "title": f"API compatibility causing data transformations",
                "details": f"Methods {', '.join(overlap)} require both API compatibility and data transformation"
            })
        
        # Check if any errors are related to API or data issues
        for error in error_issues:
            # Check for compatibility or transformation related errors
            error_msg = error["error_message"].lower()
            if "compatibility" in error_msg or "attribute" in error_msg or "transform" in error_msg:
                self.findings.append({
                    "dimension": "cross",
                    "severity": "high",
                    "title": f"Errors likely caused by compatibility issues",
                    "details": f"Error {error['error_type']} may be related to API/data compatibility issues"
                })
    
    def get_findings(self, min_severity: str = "low") -> List[Dict[str, Any]]:
        """
        Get analysis findings.
        
        Args:
            min_severity: Minimum severity level ("low", "medium", "high")
            
        Returns:
            List of findings
        """
        severity_levels = {
            "low": 1,
            "medium": 2,
            "high": 3
        }
        
        min_level = severity_levels.get(min_severity.lower(), 1)
        
        return [
            finding for finding in self.findings
            if severity_levels.get(finding["severity"], 0) >= min_level
        ]
    
    def get_dimension_data(self, dimension: str) -> Dict[str, Any]:
        """
        Get analysis data for a specific dimension.
        
        Args:
            dimension: Dimension name
            
        Returns:
            Dimension analysis data
        """
        return self.dimension_analyses.get(dimension, {})
    
    def visualize_dimension(self, dimension: str):
        """
        Visualize a specific dimension.
        
        Args:
            dimension: Dimension to visualize
        """
        if not VISUALIZE_AVAILABLE:
            logger.error("Visualization libraries not available")
            return
        
        if dimension == "api":
            self._visualize_api_dimension()
        elif dimension == "data":
            self._visualize_data_dimension()
        elif dimension == "error":
            self._visualize_error_dimension()
        elif dimension == "flow":
            self._visualize_flow_dimension()
        elif dimension == "concurrency":
            self._visualize_concurrency_dimension()
        else:
            logger.error(f"Unknown dimension: {dimension}")
    
    def _visualize_api_dimension(self):
        """Visualize API dimension."""
        api_issues = self.dimension_analyses["api"].get("issues", [])
        
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes for components and methods
        components = set()
        methods = set()
        
        for issue in api_issues:
            component = issue["component"]
            method = issue["method"]
            components.add(component)
            methods.add(method)
            
            # Add edge from component to method
            G.add_edge(component, method)
        
        # Create position layout
        pos = nx.spring_layout(G)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Draw component nodes
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=list(components),
                              node_color='lightblue',
                              node_size=700,
                              alpha=0.8)
        
        # Draw method nodes
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=list(methods),
                              node_color='lightgreen',
                              node_size=500,
                              alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
        
        # Add labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
        
        plt.title("API Compatibility Issues")
        plt.axis('off')
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"api_dimension_{timestamp}.png"
        plt.savefig(filename)
        logger.info(f"API visualization saved to {filename}")
        
        # Show figure
        plt.show()
    
    def _visualize_error_dimension(self):
        """Visualize error dimension."""
        common_errors = self.dimension_analyses["error"].get("common_errors", [])
        
        # Check if we have data to visualize
        if not common_errors:
            logger.error("No error data available for visualization")
            return
        
        # Prepare data for bar chart
        errors = [f"{e['error_type']}" for e in common_errors[:5]]
        counts = [e["count"] for e in common_errors[:5]]
        
        # Create bar chart
        plt.figure(figsize=(12, 6))
        bars = plt.bar(errors, counts, color='salmon')
        
        # Add count labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.title("Most Common Errors")
        plt.xlabel("Error Type")
        plt.ylabel("Occurrences")
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"error_dimension_{timestamp}.png"
        plt.savefig(filename)
        logger.info(f"Error visualization saved to {filename}")
        
        # Show figure
        plt.show()
    
    def _visualize_flow_dimension(self):
        """Visualize flow dimension."""
        call_paths = self.dimension_analyses["flow"].get("call_paths", {})
        
        # Check if we have data to visualize
        if not call_paths:
            logger.error("No flow data available for visualization")
            return
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add edges for call paths
        for tx_id, path_data in call_paths.items():
            path = path_data.get("path", [])
            component = path_data.get("component", "unknown")
            
            # Add consecutive method calls as edges
            for i in range(len(path) - 1):
                G.add_edge(path[i], path[i+1])
        
        # Create position layout
        pos = nx.spring_layout(G)
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                              node_color='lightgreen',
                              node_size=500,
                              alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, 
                              edge_color='gray', arrows=True)
        
        # Add labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
        
        plt.title("Execution Flow")
        plt.axis('off')
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"flow_dimension_{timestamp}.png"
        plt.savefig(filename)
        logger.info(f"Flow visualization saved to {filename}")
        
        # Show figure
        plt.show()
    
    def _visualize_data_dimension(self):
        """Visualize data dimension."""
        data_issues = self.dimension_analyses["data"].get("issues", [])
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes and edges for data transformations
        for issue in data_issues:
            if issue["issue"] == "Data structure transformation":
                # Extract from-to from transformation
                transform = issue.get("transformation", "unknown → unknown")
                from_type, to_type = transform.split(" → ")
                
                # Add nodes and edge
                G.add_node(from_type, type="data")
                G.add_node(to_type, type="data")
                G.add_edge(from_type, to_type, component=issue.get("component", "unknown"))
        
        # Check if graph is empty
        if not G.nodes():
            logger.error("No data transformation information available for visualization")
            return
        
        # Create position layout
        pos = nx.spring_layout(G)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                              node_color='lightblue',
                              node_size=700,
                              alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=2.0, alpha=0.7,
                              edge_color='red', arrows=True)
        
        # Add labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
        
        # Add edge labels (component names)
        edge_labels = {(u, v): d["component"] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title("Data Transformations")
        plt.axis('off')
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data_dimension_{timestamp}.png"
        plt.savefig(filename)
        logger.info(f"Data visualization saved to {filename}")
        
        # Show figure
        plt.show()
    
    def _visualize_concurrency_dimension(self):
        """Visualize concurrency dimension."""
        system_events = self.report_data.get("system_events", [])
        
        # Filter for relevant events
        relevant_events = [e for e in system_events 
                          if e.get("type") in ["component_initialize", "system_start", "system_initialized"]]
        
        # Check if we have enough data
        if len(relevant_events) < 2:
            logger.error("Not enough event data for concurrency visualization")
            return
        
        # Sort by timestamp
        relevant_events.sort(key=lambda e: e.get("timestamp", ""))
        
        # Extract timestamps as datetime objects
        timestamps = []
        labels = []
        colors = []
        
        for event in relevant_events:
            try:
                timestamp = datetime.fromisoformat(event.get("timestamp", ""))
                event_type = event.get("type", "unknown")
                component = event.get("data", {}).get("component", "system")
                
                timestamps.append(timestamp)
                labels.append(f"{event_type}: {component}")
                
                # Color by event type
                if "system" in event_type:
                    colors.append('green')
                else:
                    colors.append('blue')
                
            except (ValueError, TypeError):
                continue
        
        # Convert to seconds from first event
        if timestamps:
            start_time = timestamps[0]
            seconds = [(t - start_time).total_seconds() for t in timestamps]
            
            # Create timeline visualization
            plt.figure(figsize=(12, 6))
            
            # Plot events
            for i, (time_offset, label, color) in enumerate(zip(seconds, labels, colors)):
                plt.plot([time_offset, time_offset], [0, 1], color=color, alpha=0.7, linewidth=2)
                plt.text(time_offset, 1.01, label, rotation=45, ha='right', fontsize=8)
            
            plt.title("Component Initialization Timeline")
            plt.xlabel("Seconds from start")
            plt.yticks([])
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"concurrency_dimension_{timestamp}.png"
            plt.savefig(filename)
            logger.info(f"Concurrency visualization saved to {filename}")
            
            # Show figure
            plt.show()
        else:
            logger.error("No valid timestamps found for concurrency visualization")

def print_findings(analyzer, min_severity="low"):
    """Print analysis findings."""
    findings = analyzer.get_findings(min_severity)
    
    if not findings:
        print(f"\nNo findings with severity >= {min_severity}")
        return
    
    print(f"\n{'='*80}")
    print(f"DARF Analysis Findings (severity >= {min_severity})")
    print(f"{'='*80}")
    
    for i, finding in enumerate(findings, 1):
        severity = finding["severity"].upper()
        title = finding["title"]
        dimension = finding["dimension"].upper()
        details = finding["details"]
        
        print(f"\n{i}. [{severity}] {title} ({dimension})")
        print(f"   {details}")
    
    print(f"\n{'='*80}")

def main():
    parser = argparse.ArgumentParser(description="DARF Multi-Dimensional Analysis Tool")
    parser.add_argument("--report-path", help="Path to DARF debug report")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--dimensions", help="Comma-separated list of dimensions to analyze (api,data,error,flow,concurrency)")
    parser.add_argument("--min-severity", default="low", choices=["low", "medium", "high"], help="Minimum severity of findings to report")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = DimensionalAnalyzer(args.report_path)
    
    # Determine dimensions to analyze
    dimensions = []
    if args.dimensions:
        dimensions = [d.strip() for d in args.dimensions.split(",")]
    else:
        # Analyze all by default
        dimensions = ["api", "data", "error", "flow", "concurrency"]
    
    # Run analysis for each dimension
    for dimension in dimensions:
        if dimension == "api":
            analyzer.analyze_api_dimension()
        elif dimension == "data":
            analyzer.analyze_data_dimension()
        elif dimension == "error":
            analyzer.analyze_error_dimension()
        elif dimension == "flow":
            analyzer.analyze_flow_dimension()
        elif dimension == "concurrency":
            analyzer.analyze_concurrency_dimension()
        else:
            logger.warning(f"Unknown dimension: {dimension}")
    
    # Correlate dimensions
    analyzer._correlate_dimensions()
    
    # Print findings
    print_findings(analyzer, args.min_severity)
    
    # Visualize if requested
    if args.visualize:
        if not VISUALIZE_AVAILABLE:
            logger.error("Visualization libraries not available. Install with: pip install matplotlib networkx")
        else:
            for dimension in dimensions:
                analyzer.visualize_dimension(dimension)

if __name__ == "__main__":
    main()
