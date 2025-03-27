#!/usr/bin/env python3
"""
DARF Framework Stress Test Script

This script puts the DARF framework under high load by testing:
1. Event bus processing capabilities
2. Concurrent operations
3. System stability under load
4. Knowledge graph query performance
"""

import asyncio
import time
import random
import logging
import multiprocessing
import json
import argparse
import sys
from datetime import datetime
from typing import Dict, Any, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/stress_test_{time.strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

logger = logging.getLogger("DARF.StressTest")

# Import DARF components
from src.core.darf_system import DARFSystem
from src.modules.knowledge_graph.knowledge_graph_component import KnowledgeGraphComponent
from src.modules.event_bus.improved_event_bus_component import ImprovedEventBusComponent
from src.types.common_types import Result
from src.utils.concurrency import ensure_async, event_loop_manager


class StressTestHandler:
    """Handler for stress test events."""
    
    def __init__(self, component_id: str):
        """Initialize the handler."""
        self.component_id = component_id
        self.processed_count = 0
        self.start_time = time.time()
        self.processing_times = []
    
    async def handle_event(self, event_type: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an event."""
        start_time = time.time()
        
        # Simulate some processing time
        await asyncio.sleep(random.uniform(0.001, 0.01))
        
        # Occasionally introduce random delays to simulate complex processing
        if random.random() < 0.05:
            await asyncio.sleep(random.uniform(0.05, 0.2))
        
        self.processed_count += 1
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        return {
            "success": True,
            "component_id": self.component_id,
            "processed_count": self.processed_count,
            "processing_time": processing_time
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics."""
        if not self.processing_times:
            return {
                "component_id": self.component_id,
                "processed_count": self.processed_count,
                "avg_processing_time": 0,
                "min_processing_time": 0,
                "max_processing_time": 0,
                "total_runtime": time.time() - self.start_time
            }
            
        return {
            "component_id": self.component_id,
            "processed_count": self.processed_count,
            "avg_processing_time": sum(self.processing_times) / len(self.processing_times),
            "min_processing_time": min(self.processing_times),
            "max_processing_time": max(self.processing_times),
            "total_runtime": time.time() - self.start_time
        }


class DARFStressTest:
    """DARF Framework stress test."""
    
    def __init__(self, num_events: int = 10000, num_handlers: int = 10, event_batch_size: int = 100):
        """
        Initialize the stress test.
        
        Args:
            num_events: Number of events to publish
            num_handlers: Number of event handlers to register
            event_batch_size: Number of events to publish in each batch
        """
        self.num_events = num_events
        self.num_handlers = num_handlers
        self.event_batch_size = event_batch_size
        self.config = self._load_config()
        self.system = None
        self.event_bus = None
        self.knowledge_graph = None
        self.handlers = []
        self.published_count = 0
        self.start_time = None
        self.end_time = None
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration."""
        try:
            with open("config/standard.json", "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return {
                "system": {
                    "name": "DARF",
                    "version": "1.0.0"
                },
                "modes": {
                    "standard": {
                        "components": ["knowledge_graph", "event_bus"],
                        "web_port": 5000,
                        "frontend_port": 3000
                    }
                },
                "components": {
                    "knowledge_graph": {
                        "enabled": True,
                        "data_path": "datasets/processed",
                        "auto_load": True
                    },
                    "event_bus": {
                        "enabled": True,
                        "high_availability": True,
                        "metrics_enabled": True
                    }
                }
            }
    
    async def setup(self) -> bool:
        """Set up the stress test."""
        logger.info("Setting up DARF stress test")
        
        try:
            # Create DARF system
            self.system = DARFSystem(self.config)
            
            # Create and register components
            self.knowledge_graph = KnowledgeGraphComponent("knowledge_graph", self.config.get("components", {}).get("knowledge_graph", {}))
            await self.system.register_component("knowledge_graph", self.knowledge_graph)
            
            self.event_bus = ImprovedEventBusComponent("event_bus", self.config.get("components", {}).get("event_bus", {}))
            await self.system.register_component("event_bus", self.event_bus)
            
            # Start the system
            result = await self.system.start()
            if not result.has_value() or not result.value:
                logger.error(f"Failed to start DARF system: {result.error if result.has_error() else 'Unknown error'}")
                return False
            
            # Create and register handlers
            for i in range(self.num_handlers):
                handler = StressTestHandler(f"handler_{i}")
                self.handlers.append(handler)
                
                # Register handler for multiple event types
                event_types = [
                    f"stress.test.event.{i}",
                    "stress.test.event.all",
                    f"stress.test.priority.{i % 4}"
                ]
                
                for event_type in event_types:
                    self.event_bus.subscribe(
                        event_type, 
                        ensure_async(handler.handle_event),
                        {"max_retries": 2}
                    )
            
            logger.info(f"Registered {self.num_handlers} handlers")
            
            # Add some test data to the knowledge graph
            for i in range(100):
                self.knowledge_graph.add_node(
                    f"test_node_{i}",
                    {
                        "type": "test",
                        "value": i,
                        "created_at": datetime.now().isoformat()
                    }
                )
                
                if i > 0:
                    self.knowledge_graph.add_relationship(
                        f"test_node_{i}", 
                        f"test_node_{i-1}", 
                        "related_to"
                    )
            
            logger.info("Added test data to knowledge graph")
            
            return True
        except Exception as e:
            logger.error(f"Error setting up stress test: {e}", exc_info=True)
            return False
    
    async def run(self) -> Dict[str, Any]:
        """Run the stress test."""
        logger.info(f"Starting stress test with {self.num_events} events and {self.num_handlers} handlers")
        
        self.start_time = time.time()
        
        try:
            # Run event publishing test
            await self._run_event_test()
            
            # Run knowledge graph query test
            await self._run_knowledge_graph_test()
            
            # Run concurrent operation test
            await self._run_concurrent_test()
            
        except Exception as e:
            logger.error(f"Error running stress test: {e}", exc_info=True)
        finally:
            self.end_time = time.time()
            
            # Get results
            results = await self._get_results()
            
            # Shut down the system
            await self.system.stop()
            
            return results
    
    async def _run_event_test(self) -> None:
        """Run event publishing test."""
        logger.info("Running event publishing test")
        
        # Publish events in batches
        for batch in range(0, self.num_events, self.event_batch_size):
            batch_size = min(self.event_batch_size, self.num_events - batch)
            batch_start = time.time()
            
            # Create and publish events
            tasks = []
            for i in range(batch_size):
                event_index = batch + i
                
                # Create event with random properties
                event = {
                    "type": f"stress.test.event.{random.randint(0, self.num_handlers - 1)}",
                    "data": {
                        "id": str(event_index),
                        "value": random.random(),
                        "timestamp": time.time(),
                        "payload": {
                            "field1": random.randint(1, 1000),
                            "field2": f"test_{event_index}",
                            "field3": [random.random() for _ in range(5)]
                        }
                    }
                }
                
                # Occasionally create high-priority events
                if random.random() < 0.1:
                    event["priority"] = "HIGH"
                
                # Occasionally create events that all handlers will process
                if random.random() < 0.05:
                    event["type"] = "stress.test.event.all"
                
                # Publish event
                tasks.append(self.event_bus.publish(event))
                self.published_count += 1
            
            # Wait for all events to be published
            await asyncio.gather(*tasks)
            
            batch_time = time.time() - batch_start
            logger.info(f"Published batch {batch // self.event_batch_size + 1}: {batch_size} events in {batch_time:.2f}s")
            
            # Allow some time for event processing
            await asyncio.sleep(0.1)
    
    async def _run_knowledge_graph_test(self) -> None:
        """Run knowledge graph query test."""
        logger.info("Running knowledge graph query test")
        
        # Run multiple queries in parallel
        tasks = []
        for i in range(100):
            tasks.append(self._run_single_query(i))
        
        await asyncio.gather(*tasks)
    
    async def _run_single_query(self, index: int) -> None:
        """Run a single query."""
        query_type = index % 4
        
        if query_type == 0:
            # Get node by ID
            node_id = f"test_node_{random.randint(0, 99)}"
            node = self.knowledge_graph.get_node(node_id)
            
        elif query_type == 1:
            # Get relationships
            node_id = f"test_node_{random.randint(0, 99)}"
            relationships = self.knowledge_graph.get_relationships(node_id)
            
        elif query_type == 2:
            # Search by property
            value = random.randint(0, 99)
            nodes = self.knowledge_graph.search_nodes(lambda n: n.get("value") == value)
            
        else:
            # Get all nodes of type
            nodes = self.knowledge_graph.search_nodes(lambda n: n.get("type") == "test")
    
    async def _run_concurrent_test(self) -> None:
        """Run concurrent operation test."""
        logger.info("Running concurrent operation test")
        
        # Run multiple operations concurrently
        tasks = []
        for i in range(200):
            # Mix of different operations
            op_type = i % 5
            
            if op_type == 0:
                # Add node
                node_id = f"concurrent_node_{i}"
                tasks.append(asyncio.create_task(
                    asyncio.to_thread(
                        self.knowledge_graph.add_node,
                        node_id,
                        {
                            "type": "concurrent",
                            "value": i,
                            "created_at": datetime.now().isoformat()
                        }
                    )
                ))
                
            elif op_type == 1:
                # Add relationship if both nodes exist
                source = f"concurrent_node_{i-1}" if i > 0 else f"test_node_{random.randint(0, 99)}"
                target = f"concurrent_node_{i-2}" if i > 1 else f"test_node_{random.randint(0, 99)}"
                tasks.append(asyncio.create_task(
                    asyncio.to_thread(
                        self.knowledge_graph.add_relationship,
                        source, 
                        target,
                        "concurrent_related"
                    )
                ))
                
            elif op_type == 2:
                # Publish event
                event = {
                    "type": f"stress.test.priority.{i % 4}",
                    "data": {
                        "id": f"concurrent_{i}",
                        "value": random.random(),
                        "timestamp": time.time()
                    }
                }
                tasks.append(self.event_bus.publish(event))
                self.published_count += 1
                
            elif op_type == 3:
                # Query knowledge graph
                tasks.append(self._run_single_query(i))
                
            else:
                # System status check
                tasks.append(asyncio.create_task(
                    asyncio.to_thread(
                        self.system.get_status
                    )
                ))
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
    
    async def _get_results(self) -> Dict[str, Any]:
        """Get test results."""
        # Wait a bit to allow event processing to complete
        await asyncio.sleep(2)
        
        # Get system status
        status = self.system.get_status()
        
        # Get handler statistics
        handler_stats = [handler.get_stats() for handler in self.handlers]
        
        # Get event bus statistics
        eb_status = status["component_statuses"]["event_bus"]
        
        # Calculate statistics
        total_processed = sum(handler.processed_count for handler in self.handlers)
        runtime = self.end_time - self.start_time
        events_per_second = self.published_count / runtime
        
        # Compile results
        results = {
            "duration_seconds": runtime,
            "published_events": self.published_count,
            "processed_events": total_processed,
            "events_per_second": events_per_second,
            "handlers": len(self.handlers),
            "event_bus_metrics": {
                "published_count": eb_status.get("published_count", 0),
                "processed_count": eb_status.get("processed_count", 0),
                "failed_count": eb_status.get("failed_count", 0),
                "retry_count": eb_status.get("retry_count", 0),
                "queue_sizes": eb_status.get("queue_sizes", {})
            },
            "handler_stats": handler_stats,
            "system_metrics": {
                "component_starts": status["metrics"]["component_starts"],
                "component_stops": status["metrics"]["component_stops"],
                "successful_operations": status["metrics"]["successful_operations"],
                "failed_operations": status["metrics"]["failed_operations"]
            }
        }
        
        return results


async def run_stress_test(args: argparse.Namespace) -> None:
    """Run the stress test."""
    # Create and run stress test
    test = DARFStressTest(
        num_events=args.events,
        num_handlers=args.handlers,
        event_batch_size=args.batch_size
    )
    
    # Set up the test
    if not await test.setup():
        logger.error("Failed to set up stress test")
        return
    
    # Run the test
    results = await test.run()
    
    # Print results
    logger.info("\n" + "=" * 50)
    logger.info("DARF STRESS TEST RESULTS")
    logger.info("=" * 50)
    logger.info(f"Duration: {results['duration_seconds']:.2f} seconds")
    logger.info(f"Published events: {results['published_events']}")
    logger.info(f"Processed events: {results['processed_events']}")
    logger.info(f"Events per second: {results['events_per_second']:.2f}")
    logger.info(f"Number of handlers: {results['handlers']}")
    logger.info("\nEvent Bus Metrics:")
    logger.info(f"- Published: {results['event_bus_metrics']['published_count']}")
    logger.info(f"- Processed: {results['event_bus_metrics']['processed_count']}")
    logger.info(f"- Failed: {results['event_bus_metrics']['failed_count']}")
    logger.info(f"- Retries: {results['event_bus_metrics']['retry_count']}")
    
    # Print handler statistics summary
    processed_counts = [h["processed_count"] for h in results["handler_stats"]]
    avg_times = [h["avg_processing_time"] for h in results["handler_stats"] if h["processed_count"] > 0]
    
    if processed_counts:
        logger.info("\nHandler Statistics:")
        logger.info(f"- Min processed: {min(processed_counts)}")
        logger.info(f"- Max processed: {max(processed_counts)}")
        logger.info(f"- Avg processed: {sum(processed_counts) / len(processed_counts):.2f}")
    
    if avg_times:
        logger.info(f"- Min avg time: {min(avg_times):.6f} seconds")
        logger.info(f"- Max avg time: {max(avg_times):.6f} seconds")
        logger.info(f"- Overall avg time: {sum(avg_times) / len(avg_times):.6f} seconds")
    
    # Save results to file
    results_file = f"metrics/stress_test_{time.strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("metrics", exist_ok=True)
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nDetailed results saved to {results_file}")
    logger.info("=" * 50)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="DARF Framework Stress Test")
    parser.add_argument("--events", type=int, default=10000, help="Number of events to publish")
    parser.add_argument("--handlers", type=int, default=10, help="Number of event handlers to register")
    parser.add_argument("--batch-size", type=int, default=100, help="Event batch size")
    
    args = parser.parse_args()
    
    # Set up asyncio policy for Windows if needed
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Run stress test
    asyncio.run(run_stress_test(args))
