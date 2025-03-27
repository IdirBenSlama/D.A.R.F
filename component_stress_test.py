#!/usr/bin/env python3
"""
DARF Component-Specific Stress Test Script

This script allows focused stress testing of individual DARF components:
- Event Bus stress test
- Knowledge Graph stress test 
- System stability test
"""

import asyncio
import time
import random
import logging
import json
import argparse
import sys
import os
from datetime import datetime
from typing import Dict, Any, List, Callable, Awaitable

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/component_test_{time.strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

logger = logging.getLogger("DARF.ComponentTest")

# Import DARF components
from src.core.darf_system import DARFSystem
from src.modules.knowledge_graph.knowledge_graph_component import KnowledgeGraphComponent
from src.modules.event_bus.improved_event_bus_component import ImprovedEventBusComponent
from src.types.common_types import Result
from src.utils.concurrency import ensure_async, event_loop_manager


class EventBusStressTest:
    """Test the Event Bus component under load."""
    
    def __init__(self, event_bus, num_events=5000, num_topics=10, batch_size=100):
        """Initialize the test."""
        self.event_bus = event_bus
        self.num_events = num_events
        self.num_topics = num_topics
        self.batch_size = batch_size
        self.handlers = {}
        self.processed = 0
        self.published = 0
        self.failed = 0
        
    async def setup(self):
        """Set up the test."""
        logger.info("Setting up Event Bus stress test")
        
        # Register handlers for each topic
        for i in range(self.num_topics):
            topic = f"test.topic.{i}"
            
            # Create handler
            async def handler(event_type, event_data, topic_id=i):
                self.processed += 1
                # Simulate processing
                await asyncio.sleep(random.uniform(0.001, 0.01))
                return {"success": True, "topic": topic_id}
            
            # Register handler
            result = self.event_bus.subscribe(topic, handler)
            if result.has_error():
                logger.error(f"Failed to register handler for topic {topic}: {result.error}")
                return False
                
            self.handlers[topic] = handler
            
        # Also register a wildcard handler
        async def wildcard_handler(event_type, event_data):
            # Just count it, don't increment processed
            await asyncio.sleep(0.001)
            return {"success": True, "wildcard": True}
            
        result = self.event_bus.subscribe("test.topic.*", wildcard_handler)
        if result.has_error():
            logger.error(f"Failed to register wildcard handler: {result.error}")
            
        logger.info(f"Registered {len(self.handlers)} topic handlers")
        return True
        
    async def run(self):
        """Run the test."""
        logger.info(f"Running Event Bus stress test with {self.num_events} events")
        start_time = time.time()
        
        # Publish events in batches
        for batch_start in range(0, self.num_events, self.batch_size):
            batch_size = min(self.batch_size, self.num_events - batch_start)
            
            # Create publishing tasks
            tasks = []
            for i in range(batch_size):
                # Select random topic
                topic_id = random.randint(0, self.num_topics - 1)
                topic = f"test.topic.{topic_id}"
                
                # Create event
                event = {
                    "type": topic,
                    "data": {
                        "id": f"event_{batch_start + i}",
                        "value": random.random(),
                        "timestamp": time.time()
                    }
                }
                
                # Occasionally use high priority
                if random.random() < 0.1:
                    event["priority"] = "HIGH"
                
                # Publish event
                tasks.append(self.event_bus.publish(event))
                self.published += 1
            
            # Wait for all publishing tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count failures
            for result in results:
                if isinstance(result, Exception) or (hasattr(result, 'has_error') and result.has_error()):
                    self.failed += 1
            
            # Log progress
            if (batch_start // self.batch_size) % 10 == 0:
                logger.info(f"Published {batch_start + batch_size}/{self.num_events} events")
                
            # Allow time for processing between batches
            await asyncio.sleep(0.05)
        
        # Give time for events to be processed
        logger.info("Waiting for event processing to complete...")
        await asyncio.sleep(2)
        
        # Calculate results
        duration = time.time() - start_time
        
        # Get status from event bus
        status = {}
        if hasattr(self.event_bus, 'get_status'):
            status = self.event_bus.get_status()
        
        return {
            "duration_seconds": duration,
            "events_per_second": self.published / duration,
            "published": self.published,
            "processed": self.processed,
            "failed": self.failed,
            "event_bus_status": status
        }


class KnowledgeGraphStressTest:
    """Test the Knowledge Graph component under load."""
    
    def __init__(self, kg, num_nodes=1000, num_relationships=2000, batch_size=100):
        """Initialize the test."""
        self.kg = kg
        self.num_nodes = num_nodes
        self.num_relationships = num_relationships
        self.batch_size = batch_size
        self.nodes_created = 0
        self.relationships_created = 0
        self.failed_operations = 0
        
    async def setup(self):
        """Set up the test."""
        logger.info("Setting up Knowledge Graph stress test")
        return True
        
    async def run(self):
        """Run the test."""
        logger.info(f"Running Knowledge Graph stress test with {self.num_nodes} nodes and {self.num_relationships} relationships")
        start_time = time.time()
        
        # Create nodes
        await self._create_nodes()
        
        # Create relationships
        await self._create_relationships()
        
        # Run queries
        query_results = await self._run_queries()
        
        # Calculate results
        duration = time.time() - start_time
        
        # Get status from knowledge graph
        status = {}
        if hasattr(self.kg, 'get_status'):
            status = self.kg.get_status()
        
        return {
            "duration_seconds": duration,
            "nodes_per_second": self.nodes_created / duration,
            "relationships_per_second": self.relationships_created / duration,
            "nodes_created": self.nodes_created,
            "relationships_created": self.relationships_created,
            "failed_operations": self.failed_operations,
            "query_results": query_results,
            "knowledge_graph_status": status
        }
    
    async def _create_nodes(self):
        """Create nodes in the knowledge graph."""
        logger.info(f"Creating {self.num_nodes} nodes")
        
        # Create nodes in batches
        for batch_start in range(0, self.num_nodes, self.batch_size):
            batch_size = min(self.batch_size, self.num_nodes - batch_start)
            
            # Create tasks for node creation
            tasks = []
            for i in range(batch_size):
                node_id = f"stress_node_{batch_start + i}"
                node_data = {
                    "id": node_id,
                    "type": f"type_{(batch_start + i) % 10}",
                    "value": random.random(),
                    "timestamp": time.time(),
                    "properties": {
                        "prop1": random.randint(1, 100),
                        "prop2": f"value_{(batch_start + i) % 20}"
                    }
                }
                
                # Use to_thread since add_node might be a blocking operation
                tasks.append(asyncio.to_thread(
                    self._safe_add_node, node_id, node_data
                ))
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successful operations
            for result in results:
                if result is True:
                    self.nodes_created += 1
                else:
                    self.failed_operations += 1
            
            # Log progress
            if (batch_start // self.batch_size) % 5 == 0:
                logger.info(f"Created {self.nodes_created}/{self.num_nodes} nodes")
    
    def _safe_add_node(self, node_id, node_data):
        """Safely add a node, catching exceptions."""
        try:
            self.kg.add_node(node_id, node_data)
            return True
        except Exception as e:
            logger.error(f"Error adding node {node_id}: {e}")
            return False
    
    async def _create_relationships(self):
        """Create relationships in the knowledge graph."""
        logger.info(f"Creating {self.num_relationships} relationships")
        
        # Create relationships in batches
        for batch_start in range(0, self.num_relationships, self.batch_size):
            batch_size = min(self.batch_size, self.num_relationships - batch_start)
            
            # Create tasks for relationship creation
            tasks = []
            for i in range(batch_size):
                # Randomly select source and target nodes
                source_id = f"stress_node_{random.randint(0, max(0, self.nodes_created - 1))}"
                target_id = f"stress_node_{random.randint(0, max(0, self.nodes_created - 1))}"
                
                # Skip self-relationships occasionally
                if source_id == target_id and random.random() < 0.5:
                    continue
                    
                # Random relationship type
                rel_type = f"rel_type_{random.randint(0, 5)}"
                
                # Use to_thread since add_relationship might be a blocking operation
                tasks.append(asyncio.to_thread(
                    self._safe_add_relationship, source_id, target_id, rel_type
                ))
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successful operations
            for result in results:
                if result is True:
                    self.relationships_created += 1
                else:
                    self.failed_operations += 1
            
            # Log progress
            if (batch_start // self.batch_size) % 5 == 0:
                logger.info(f"Created {self.relationships_created}/{self.num_relationships} relationships")
    
    def _safe_add_relationship(self, source_id, target_id, rel_type):
        """Safely add a relationship, catching exceptions."""
        try:
            self.kg.add_relationship(source_id, target_id, rel_type)
            return True
        except Exception as e:
            logger.error(f"Error adding relationship {source_id}->{target_id}: {e}")
            return False
    
    async def _run_queries(self):
        """Run various queries on the knowledge graph."""
        logger.info("Running knowledge graph queries")
        
        query_results = {
            "get_node": {"success": 0, "failed": 0, "time": 0},
            "get_relationships": {"success": 0, "failed": 0, "time": 0},
            "search_by_property": {"success": 0, "failed": 0, "time": 0},
            "search_by_type": {"success": 0, "failed": 0, "time": 0}
        }
        
        # Run get_node queries
        start_time = time.time()
        for i in range(100):
            node_id = f"stress_node_{random.randint(0, max(0, self.nodes_created - 1))}"
            try:
                node = self.kg.get_node(node_id)
                if node:
                    query_results["get_node"]["success"] += 1
                else:
                    query_results["get_node"]["failed"] += 1
            except Exception:
                query_results["get_node"]["failed"] += 1
        query_results["get_node"]["time"] = time.time() - start_time
        
        # Run get_relationships queries
        start_time = time.time()
        for i in range(100):
            node_id = f"stress_node_{random.randint(0, max(0, self.nodes_created - 1))}"
            try:
                rels = self.kg.get_relationships(node_id)
                if rels is not None:  # May return empty list
                    query_results["get_relationships"]["success"] += 1
                else:
                    query_results["get_relationships"]["failed"] += 1
            except Exception:
                query_results["get_relationships"]["failed"] += 1
        query_results["get_relationships"]["time"] = time.time() - start_time
        
        # Run search_by_property (if supported)
        if hasattr(self.kg, 'search_nodes'):
            start_time = time.time()
            for i in range(20):
                prop_value = random.randint(1, 100)
                try:
                    # Use lambda to search for property value
                    nodes = self.kg.search_nodes(
                        lambda n: n.get("properties", {}).get("prop1") == prop_value
                    )
                    if nodes is not None:  # May return empty list
                        query_results["search_by_property"]["success"] += 1
                    else:
                        query_results["search_by_property"]["failed"] += 1
                except Exception:
                    query_results["search_by_property"]["failed"] += 1
            query_results["search_by_property"]["time"] = time.time() - start_time
            
            # Run search_by_type
            start_time = time.time()
            for i in range(10):
                type_value = f"type_{random.randint(0, 9)}"
                try:
                    # Use lambda to search for type
                    nodes = self.kg.search_nodes(
                        lambda n: n.get("type") == type_value
                    )
                    if nodes is not None:  # May return empty list
                        query_results["search_by_type"]["success"] += 1
                    else:
                        query_results["search_by_type"]["failed"] += 1
                except Exception:
                    query_results["search_by_type"]["failed"] += 1
            query_results["search_by_type"]["time"] = time.time() - start_time
        
        logger.info(f"Completed {sum(q['success'] + q['failed'] for q in query_results.values())} queries")
        return query_results


class SystemStabilityTest:
    """Test overall system stability under load."""
    
    def __init__(self, system, num_operations=1000, duration_seconds=60):
        """Initialize the test."""
        self.system = system
        self.num_operations = num_operations
        self.duration_seconds = duration_seconds
        self.components = {}
        self.operation_results = {
            "success": 0,
            "failed": 0,
            "by_type": {}
        }
        
    async def setup(self):
        """Set up the test by getting references to all components."""
        logger.info("Setting up System stability test")
        
        # Get all components
        status = self.system.get_status()
        for component_name in status.get("components", []):
            self.components[component_name] = self.system.get_component(component_name)
            
        logger.info(f"Found {len(self.components)} components")
        return True
        
    async def run(self):
        """Run the test for the specified duration or number of operations."""
        logger.info(f"Running System stability test for up to {self.duration_seconds}s with max {self.num_operations} operations")
        start_time = time.time()
        operations_completed = 0
        
        # Run until duration or operation limit is reached
        while (time.time() - start_time < self.duration_seconds and 
               operations_completed < self.num_operations):
            
            # Pick random operation type
            operation_type = random.choice([
                "status_check", 
                "component_status",
                "component_operation"
            ])
            
            # Initialize operation type stats if needed
            if operation_type not in self.operation_results["by_type"]:
                self.operation_results["by_type"][operation_type] = {
                    "success": 0, 
                    "failed": 0
                }
            
            try:
                if operation_type == "status_check":
                    # Get system status
                    status = self.system.get_status()
                    if status and "running" in status:
                        self.operation_results["success"] += 1
                        self.operation_results["by_type"][operation_type]["success"] += 1
                    else:
                        self.operation_results["failed"] += 1
                        self.operation_results["by_type"][operation_type]["failed"] += 1
                        
                elif operation_type == "component_status":
                    # Get random component status
                    if self.components:
                        component_name = random.choice(list(self.components.keys()))
                        component = self.components[component_name]
                        
                        if hasattr(component, "get_status"):
                            status = component.get_status()
                            if status:
                                self.operation_results["success"] += 1
                                self.operation_results["by_type"][operation_type]["success"] += 1
                            else:
                                self.operation_results["failed"] += 1
                                self.operation_results["by_type"][operation_type]["failed"] += 1
                        else:
                            # No get_status method
                            self.operation_results["failed"] += 1
                            self.operation_results["by_type"][operation_type]["failed"] += 1
                    else:
                        # No components available
                        self.operation_results["failed"] += 1
                        self.operation_results["by_type"][operation_type]["failed"] += 1
                        
                elif operation_type == "component_operation":
                    # Try a component-specific operation
                    await self._run_component_operation()
                    
            except Exception as e:
                logger.error(f"Error in operation {operation_type}: {e}")
                self.operation_results["failed"] += 1
                self.operation_results["by_type"][operation_type]["failed"] += 1
            
            operations_completed += 1
            
            # Log progress
            if operations_completed % 100 == 0:
                elapsed = time.time() - start_time
                logger.info(f"Completed {operations_completed} operations in {elapsed:.2f}s")
            
            # Pause between operations
            await asyncio.sleep(0.01)
        
        # Calculate results
        duration = time.time() - start_time
        
        # Get final system status
        final_status = self.system.get_status()
        
        return {
            "duration_seconds": duration,
            "operations_completed": operations_completed,
            "operations_per_second": operations_completed / duration,
            "success_rate": self.operation_results["success"] / max(1, operations_completed),
            "operation_results": self.operation_results,
            "final_system_status": final_status
        }
    
    async def _run_component_operation(self):
        """Run a random operation on a random component."""
        if not self.components:
            self.operation_results["failed"] += 1
            if "component_operation" not in self.operation_results["by_type"]:
                self.operation_results["by_type"]["component_operation"] = {"success": 0, "failed": 0}
            self.operation_results["by_type"]["component_operation"]["failed"] += 1
            return
            
        # Select random component
        component_name = random.choice(list(self.components.keys()))
        component = self.components[component_name]
        
        # Initialize component operation stats if needed
        operation_type = f"{component_name}_operation"
        if operation_type not in self.operation_results["by_type"]:
            self.operation_results["by_type"][operation_type] = {"success": 0, "failed": 0}
        
        try:
            if component_name == "knowledge_graph":
                # Try a knowledge graph operation
                if random.random() < 0.5 and hasattr(component, "get_node"):
                    # Get a node (may not exist)
                    node_id = f"stress_node_{random.randint(0, 999)}"
                    component.get_node(node_id)
                    self.operation_results["success"] += 1
                    self.operation_results["by_type"][operation_type]["success"] += 1
                elif hasattr(component, "get_relationships"):
                    # Get relationships (may not exist)
                    node_id = f"stress_node_{random.randint(0, 999)}"
                    component.get_relationships(node_id)
                    self.operation_results["success"] += 1
                    self.operation_results["by_type"][operation_type]["success"] += 1
                else:
                    self.operation_results["failed"] += 1
                    self.operation_results["by_type"][operation_type]["failed"] += 1
                    
            elif component_name == "event_bus":
                # Try an event bus operation
                if random.random() < 0.7 and hasattr(component, "publish"):
                    # Publish an event
                    event = {
                        "type": "stress.system.test",
                        "data": {
                            "id": f"system_test_{time.time()}",
                            "value": random.random()
                        }
                    }
                    await component.publish(event)
                    self.operation_results["success"] += 1
                    self.operation_results["by_type"][operation_type]["success"] += 1
                else:
                    # Get status as fallback
                    component.get_status()
                    self.operation_results["success"] += 1
                    self.operation_results["by_type"][operation_type]["success"] += 1
                    
            else:
                # Default - try to get status or just succeed
                if hasattr(component, "get_status"):
                    component.get_status()
                self.operation_results["success"] += 1
                self.operation_results["by_type"][operation_type]["success"] += 1
                
        except Exception as e:
            logger.error(f"Error in component operation {component_name}: {e}")
            self.operation_results["failed"] += 1
            self.operation_results["by_type"][operation_type]["failed"] += 1


async def run_component_test(args):
    """Run the specified component test."""
    # Load configuration
    config = {}
    try:
        with open("config/standard.json", "r") as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        config = {
            "system": {"name": "DARF", "version": "1.0.0"},
            "modes": {"standard": {"components": ["knowledge_graph", "event_bus"]}},
            "components": {
                "knowledge_graph": {"enabled": True, "data_path": "datasets/processed"},
                "event_bus": {"enabled": True, "high_availability": True}
            }
        }
    
    # Create DARF system
    system = DARFSystem(config)
    
    # Register and start components
    if args.component in ("all", "event_bus"):
        logger.info("Registering Event Bus component")
        eb_config = config.get("components", {}).get("event_bus", {})
        event_bus = ImprovedEventBusComponent("event_bus", eb_config)
        await system.register_component("event_bus", event_bus)
    
    if args.component in ("all", "knowledge_graph"):
        logger.info("Registering Knowledge Graph component")
        kg_config = config.get("components", {}).get("knowledge_graph", {})
        knowledge_graph = KnowledgeGraphComponent("knowledge_graph", kg_config)
        await system.register_component("knowledge_graph", knowledge_graph)
    
    # Start the system
    logger.info("Starting DARF system")
    result = await system.start()
    if not result.has_value() or not result.value:
        logger.error(f"Failed to start DARF system: {result.error if result.has_error() else 'Unknown error'}")
        return
    
    results = {}
    try:
        # Run the requested test
        if args.component in ("all", "event_bus"):
            logger.info("Running Event Bus stress test")
            eb_test = EventBusStressTest(
                system.get_component("event_bus"),
                num_events=args.events,
                batch_size=args.batch_size
            )
            await eb_test.setup()
            eb_results = await eb_test.run()
            results["event_bus"] = eb_results
            
            # Print results
            logger.info("\n" + "=" * 50)
            logger.info("EVENT BUS STRESS TEST RESULTS")
            logger.info("=" * 50)
            logger.info(f"Duration: {eb_results['duration_seconds']:.2f} seconds")
            logger.info(f"Events per second: {eb_results['events_per_second']:.2f}")
            logger.info(f"Published: {eb_results['published']}")
            logger.info(f"Processed: {eb_results['processed']}")
            logger.info(f"Failed: {eb_results['failed']}")
            logger.info("=" * 50)
        
        if args.component in ("all", "knowledge_graph"):
            logger.info("Running Knowledge Graph stress test")
            kg_test = KnowledgeGraphStressTest(
                system.get_component("knowledge_graph"),
                num_nodes=args.nodes,
                num_relationships=args.relationships,
                batch_size=args.batch_size
            )
            await kg_test.setup()
            kg_results = await kg_test.run()
            results["knowledge_graph"] = kg_results
            
            # Print results
            logger.info("\n" + "=" * 50)
            logger.info("KNOWLEDGE GRAPH STRESS TEST RESULTS")
            logger.info("=" * 50)
            logger.info(f"Duration: {kg_results['duration_seconds']:.2f} seconds")
            logger.info(f"Nodes per second: {kg_results['nodes_per_second']:.2f}")
            logger.info(f"Relationships per second: {kg_results['relationships_per_second']:.2f}")
            logger.info(f"Nodes created: {kg_results['nodes_created']}")
            logger.info(f"Relationships created: {kg_results['relationships_created']}")
            logger.info(f"Failed operations: {kg_results['failed_operations']}")
            logger.info("Query performance:")
            for query_type, stats in kg_results["query_results"].items():
                if stats["success"] + stats["failed"] > 0:
                    success_rate = stats["success"] / (stats["success"] + stats["failed"])
                    logger.info(f"- {query_type}: {stats['success']}/{stats['success'] + stats['failed']} " +
                                f"({success_rate:.1%}) in {stats['time']:.3f}s")
            logger.info("=" * 50)
        
        if args.component in ("all", "system"):
            logger.info("Running System stability test")
            sys_test = SystemStabilityTest(
                system,
                num_operations=args.operations,
                duration_seconds=args.duration
            )
            await sys_test.setup()
            sys_results = await sys_test.run()
            results["system"] = sys_results
            
            # Print results
            logger.info("\n" + "=" * 50)
            logger.info("SYSTEM STABILITY TEST RESULTS")
            logger.info("=" * 50)
            logger.info(f"Duration: {sys_results['duration_seconds']:.2f} seconds")
            logger.info(f"Operations completed: {sys_results['operations_completed']}")
            logger.info(f"Operations per second: {sys_results['operations_per_second']:.2f}")
            logger.info(f"Success rate: {sys_results['success_rate']:.1%}")
            logger.info("\nOperation results by type:")
            for op_type, stats in sys_results["operation_results"]["by_type"].items():
                if stats["success"] + stats["failed"] > 0:
                    success_rate = stats["success"] / (stats["success"] + stats["failed"])
                    logger.info(f"- {op_type}: {stats['success']}/{stats['success'] + stats['failed']} " +
                                f"({success_rate:.1%})")
            logger.info("=" * 50)
            
        # Save overall results to file
        os.makedirs("metrics", exist_ok=True)
        results_file = f"metrics/component_test_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nDetailed results saved to {results_file}")
    
    finally:
        # Stop the system
        logger.info("Stopping DARF system")
        await system.stop()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="DARF Component Stress Test")
    parser.add_argument("--component", choices=["all", "event_bus", "knowledge_graph", "system"], 
                        default="all", help="Component to test")
    parser.add_argument("--events", type=int, default=5000, help="Number of events (for event bus test)")
    parser.add_argument("--nodes", type=int, default=1000, help="Number of nodes (for knowledge graph test)")
    parser.add_argument("--relationships", type=int, default=2000, 
                        help="Number of relationships (for knowledge graph test)")
    parser.add_argument("--operations", type=int, default=1000, 
                        help="Number of operations (for system test)")
    parser.add_argument("--duration", type=int, default=60, 
                        help="Maximum test duration in seconds (for system test)")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for operations")
    
    args = parser.parse_args()
    
    # Set up asyncio policy for Windows if needed
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Run the component test
    asyncio.run(run_component_test(args))
