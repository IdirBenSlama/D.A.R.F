#!/usr/bin/env python3
"""
Integration test for the Enhanced Consensus Protocol Selector.

This script tests the thread-safe, optimized protocol selector implementation
with circuit breaking, caching, and adaptive protocol selection features.
"""

import asyncio
import json
import logging
import unittest
import sys
import os
import time
import random
import uuid
from typing import List, Dict, Any, Optional
from unittest.mock import patch, AsyncMock, MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ProtocolSelectorTest")

# Import consensus modules
from src.modules.consensus_engine.protocol_selector_optimized import (
    EnhancedConsensusProtocolSelector,
    CircuitBreaker,
    CommandCache
)


class MockProtocol:
    """Mock protocol implementation for testing."""
    
    def __init__(self, name: str, fail_rate: float = 0.0, timeout_rate: float = 0.0):
        """
        Initialize mock protocol.
        
        Args:
            name: Protocol name
            fail_rate: Rate of command failures (0.0-1.0)
            timeout_rate: Rate of timeouts (0.0-1.0)
        """
        self.name = name
        self.fail_rate = fail_rate
        self.timeout_rate = timeout_rate
        self.commands_processed = 0
        self.commands_succeeded = 0
        self.commands_failed = 0
        self.commands_timeout = 0
        self.health_status = "healthy"
    
    async def submit_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Process a command with configurable success/failure rates."""
        self.commands_processed += 1
        command_id = command.get("id", str(uuid.uuid4()))
        
        # Simulate processing delay
        await asyncio.sleep(0.01)
        
        # Determine if command should timeout
        if random.random() < self.timeout_rate:
            self.commands_timeout += 1
            # Sleep longer than any reasonable timeout
            await asyncio.sleep(10.0)
            
        # Determine if command should fail
        if random.random() < self.fail_rate:
            self.commands_failed += 1
            raise Exception(f"Simulated failure in {self.name}")
            
        # Command succeeded
        self.commands_succeeded += 1
        return {
            "status": "success",
            "command_id": command_id,
            "protocol": self.name,
            "metadata": {
                "timestamp": time.time()
            }
        }
    
    async def get_health(self) -> Dict[str, Any]:
        """Get protocol health information."""
        return {
            "status": self.health_status,
            "protocol": self.name,
            "commands_processed": self.commands_processed,
            "commands_succeeded": self.commands_succeeded,
            "commands_failed": self.commands_failed,
            "commands_timeout": self.commands_timeout
        }
    
    def set_health_status(self, status: str):
        """Set protocol health status."""
        self.health_status = status


class TestCircuitBreaker(unittest.TestCase):
    """Test the CircuitBreaker implementation."""
    
    def test_initial_state(self):
        """Test initial state of circuit breaker."""
        circuit = CircuitBreaker("test", failure_threshold=3, reset_timeout=1.0)
        self.assertEqual(circuit.state, "CLOSED")
        self.assertEqual(circuit.failures, 0)
        self.assertTrue(circuit.is_allowed())
    
    def test_record_failure(self):
        """Test failure recording."""
        circuit = CircuitBreaker("test", failure_threshold=3, reset_timeout=1.0)
        
        # Record failures but stay under threshold
        circuit.record_failure()
        circuit.record_failure()
        self.assertEqual(circuit.failures, 2)
        self.assertEqual(circuit.state, "CLOSED")
        self.assertTrue(circuit.is_allowed())
        
        # Record failure that exceeds threshold
        circuit.record_failure()
        self.assertEqual(circuit.failures, 3)
        self.assertEqual(circuit.state, "OPEN")
        self.assertFalse(circuit.is_allowed())
    
    def test_record_success(self):
        """Test success recording."""
        circuit = CircuitBreaker("test", failure_threshold=2, reset_timeout=1.0)
        
        # Record failures to open circuit
        circuit.record_failure()
        circuit.record_failure()
        self.assertEqual(circuit.state, "OPEN")
        
        # Wait for reset timeout
        time.sleep(1.1)
        
        # Circuit should allow one operation in HALF_OPEN state
        self.assertTrue(circuit.is_allowed())
        self.assertEqual(circuit.state, "HALF_OPEN")
        
        # Record success to close circuit
        circuit.record_success()
        self.assertEqual(circuit.failures, 0)
        self.assertEqual(circuit.state, "CLOSED")
        self.assertTrue(circuit.is_allowed())


class TestCommandCache(unittest.TestCase):
    """Test the CommandCache implementation."""
    
    def test_get_set(self):
        """Test basic get/set operations."""
        cache = CommandCache(max_size=10, ttl=1.0)
        
        # Set and get value
        cache.set("cmd1", {"status": "success"})
        result = cache.get("cmd1")
        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "success")
        
        # Get non-existent value
        result = cache.get("cmd2")
        self.assertIsNone(result)
        
        # Wait for TTL to expire
        time.sleep(1.1)
        
        # Value should be expired
        result = cache.get("cmd1")
        self.assertIsNone(result)
    
    def test_max_size(self):
        """Test max size enforcement."""
        cache = CommandCache(max_size=3, ttl=10.0)
        
        # Add more items than max size
        cache.set("cmd1", {"status": "success"})
        cache.set("cmd2", {"status": "success"})
        cache.set("cmd3", {"status": "success"})
        cache.set("cmd4", {"status": "success"})
        
        # One of the first items should have been pruned
        count = sum(1 for x in ["cmd1", "cmd2", "cmd3", "cmd4"] if cache.get(x) is not None)
        self.assertEqual(count, 3)
    
    def test_in_progress(self):
        """Test in-progress command tracking."""
        cache = CommandCache()
        
        # Create a future
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        
        # Register command
        self.assertTrue(cache.register_in_progress("cmd1", future))
        self.assertTrue(cache.is_in_progress("cmd1"))
        
        # Try to register again (should fail)
        self.assertFalse(cache.register_in_progress("cmd1", loop.create_future()))
        
        # Complete command
        self.assertTrue(cache.complete_in_progress("cmd1", {"status": "success"}))
        self.assertTrue(future.done())
        self.assertEqual(future.result()["status"], "success")
        
        # Command should be in cache now
        self.assertFalse(cache.is_in_progress("cmd1"))
        self.assertIsNotNone(cache.get("cmd1"))


class TestEnhancedConsensusProtocolSelector(unittest.IsolatedAsyncioTestCase):
    """Test the EnhancedConsensusProtocolSelector implementation."""
    
    async def asyncSetUp(self):
        """Set up test environment."""
        # Create mock protocols
        self.hotstuff = MockProtocol("hotstuff", fail_rate=0.1)
        self.raft = MockProtocol("raft", fail_rate=0.2)
        self.cbft = MockProtocol("cbft", fail_rate=0.3)
        
        # Create mock event bus
        self.event_bus = MagicMock()
        self.event_bus.publish = AsyncMock()
        
        # Create test configuration
        self.config = {
            "byzantine_command_types": ["financial", "critical", "security"],
            "performance_command_types": ["query", "read", "status"],
            "cluster_command_types": ["local", "cluster", "internal"],
            "distributed_command_types": ["global", "cross_cluster", "network_wide"],
            "security_level_threshold": 7,
            "circuit_breaker": {
                "hotstuff_threshold": 5,
                "raft_threshold": 3,
                "cbft_threshold": 4
            },
            "cache": {
                "max_size": 100,
                "ttl": 60.0
            },
            "adaptive_selection": {
                "enabled": True
            },
            "metrics": {
                "enabled": False
            }
        }
        
        # Create protocol selector
        self.selector = EnhancedConsensusProtocolSelector(
            hotstuff=self.hotstuff,
            raft=self.raft,
            cbft=self.cbft,
            event_bus=self.event_bus,
            config=self.config
        )
        
        # Start selector
        await self.selector.start()
    
    async def asyncTearDown(self):
        """Clean up test environment."""
        await self.selector.stop()
    
    async def test_protocol_selection(self):
        """Test protocol selection logic."""
        # Financial command (byzantine) should use HotStuff
        protocol, name = await self.selector.select_protocol({
            "id": "cmd1",
            "type": "financial",
            "data": "Transfer funds"
        })
        self.assertEqual(name, "hotstuff")
        self.assertEqual(protocol, self.hotstuff)
        
        # Query command (performance) should use RAFT
        protocol, name = await self.selector.select_protocol({
            "id": "cmd2",
            "type": "query",
            "data": "Get status"
        })
        self.assertEqual(name, "raft")
        self.assertEqual(protocol, self.raft)
        
        # Global command (distributed) should use CBFT
        protocol, name = await self.selector.select_protocol({
            "id": "cmd3",
            "type": "global",
            "data": "Update global policy"
        })
        self.assertEqual(name, "cbft")
        self.assertEqual(protocol, self.cbft)
        
        # High security command should use HotStuff regardless of type
        protocol, name = await self.selector.select_protocol({
            "id": "cmd4",
            "type": "query",  # Would normally use RAFT
            "security_level": 8,  # But high security makes it use HotStuff
            "data": "Secure query"
        })
        self.assertEqual(name, "hotstuff")
        self.assertEqual(protocol, self.hotstuff)
        
        # Unknown command type should default to HotStuff
        protocol, name = await self.selector.select_protocol({
            "id": "cmd5",
            "type": "unknown",
            "data": "Unknown command"
        })
        self.assertEqual(name, "hotstuff")
        self.assertEqual(protocol, self.hotstuff)
    
    async def test_circuit_breaker_functionality(self):
        """Test circuit breaker functionality."""
        # Make RAFT fail enough times to open circuit
        for i in range(4):
            self.selector.circuit_breakers["raft"].record_failure()
        
        # Query command should now fall back to HotStuff
        protocol, name = await self.selector.select_protocol({
            "id": "cmd6",
            "type": "query",
            "data": "Get status after RAFT failure"
        })
        self.assertEqual(name, "hotstuff")
        self.assertEqual(protocol, self.hotstuff)
        
        # Record some successes to reset circuit
        for i in range(3):
            self.selector.circuit_breakers["raft"].record_success()
        
        # Query command should now use RAFT again
        protocol, name = await self.selector.select_protocol({
            "id": "cmd7",
            "type": "query",
            "data": "Get status after RAFT recovery"
        })
        self.assertEqual(name, "raft")
        self.assertEqual(protocol, self.raft)
    
    async def test_command_processing(self):
        """Test command processing flow."""
        # Process a command
        command = {
            "id": "cmd8",
            "type": "query",
            "data": "Test command processing"
        }
        result = await self.selector.process_command(command)
        
        # Result should be successful
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["command_id"], "cmd8")
        self.assertEqual(result["protocol"], "raft")
        
        # Process the same command again (should be served from cache)
        result2 = await self.selector.process_command(command)
        self.assertEqual(result2["status"], "success")
        self.assertEqual(result2["command_id"], "cmd8")
        
        # Should be the same result instance (from cache)
        self.assertIs(result, result2)
    
    async def test_fallback_logic(self):
        """Test fallback between protocols."""
        # Create a new selector with failing RAFT
        failing_raft = MockProtocol("raft", fail_rate=1.0)  # Always fails
        selector = EnhancedConsensusProtocolSelector(
            hotstuff=self.hotstuff,
            raft=failing_raft,
            cbft=self.cbft,
            config=self.config
        )
        await selector.start()
        
        try:
            # Process a query command (would normally use RAFT)
            command = {
                "id": "cmd9",
                "type": "query",
                "data": "Test fallback to HotStuff"
            }
            result = await selector.process_command(command)
            
            # Should fall back to HotStuff
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["protocol"], "hotstuff")
            self.assertTrue(result.get("fallback", False))
            
        finally:
            await selector.stop()
    
    async def test_adaptive_selection(self):
        """Test adaptive protocol selection."""
        # Create historical data that shows RAFT performs better for "special" commands
        command_history = [
            # RAFT successes
            {"id": "h1", "type": "special", "protocol": "raft", "status": "success"},
            {"id": "h2", "type": "special", "protocol": "raft", "status": "success"},
            {"id": "h3", "type": "special", "protocol": "raft", "status": "success"},
            {"id": "h4", "type": "special", "protocol": "raft", "status": "success"},
            {"id": "h5", "type": "special", "protocol": "raft", "status": "success"},
            # HotStuff failures
            {"id": "h6", "type": "special", "protocol": "hotstuff", "status": "error"},
            {"id": "h7", "type": "special", "protocol": "hotstuff", "status": "error"},
            {"id": "h8", "type": "special", "protocol": "hotstuff", "status": "error"},
        ]
        
        # Process a "special" command with history
        command = {
            "id": "cmd10",
            "type": "special",
            "data": "Test adaptive selection"
        }
        
        # This would normally go to HotStuff (default case), but should adapt to RAFT
        protocol, name = await self.selector.select_protocol(command, command_history)
        
        # Should select RAFT based on historical performance
        self.assertEqual(name, "raft")
        self.assertEqual(protocol, self.raft)
    
    async def test_health_check(self):
        """Test health check functionality."""
        # Make RAFT unhealthy
        self.raft.set_health_status("unhealthy")
        
        # Wait for health check cycle
        await asyncio.sleep(0.1)
        
        # Trigger manual health check
        await self.selector._health_check_task.__anext__()
        
        # Check status information
        status = self.selector.get_status()
        
        # Check that we have the expected data
        self.assertIn("available_protocols", status)
        self.assertIn("circuit_breakers", status)
        self.assertIn("command_cache", status)
        
        # Make all protocols healthy again
        self.hotstuff.set_health_status("healthy")
        self.raft.set_health_status("healthy")
        self.cbft.set_health_status("healthy")


async def run_concurrent_tests(selector, num_commands: int = 50):
    """
    Run concurrent command processing tests.
    
    Args:
        selector: Protocol selector to test
        num_commands: Number of concurrent commands to process
    """
    # Create commands of different types
    command_types = ["financial", "query", "global", "unknown"]
    commands = []
    
    for i in range(num_commands):
        cmd_type = random.choice(command_types)
        cmd = {
            "id": f"concurrent{i}",
            "type": cmd_type,
            "data": f"Concurrent test {i}"
        }
        commands.append(cmd)
    
    # Process commands concurrently
    start_time = time.time()
    tasks = [selector.process_command(cmd) for cmd in commands]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    end_time = time.time()
    
    # Count results by protocol and status
    protocol_counts = {"hotstuff": 0, "raft": 0, "cbft": 0, "error": 0}
    status_counts = {"success": 0, "error": 0, "fallback": 0}
    
    for result in results:
        if isinstance(result, Exception):
            protocol_counts["error"] += 1
            status_counts["error"] += 1
        else:
            protocol = result.get("protocol", "unknown")
            if protocol in protocol_counts:
                protocol_counts[protocol] += 1
            
            status = result.get("status", "unknown")
            if status in status_counts:
                status_counts[status] += 1
                
            if result.get("fallback", False):
                status_counts["fallback"] += 1
    
    logger.info(f"Processed {num_commands} commands in {end_time - start_time:.2f} seconds")
    logger.info(f"Protocol counts: {protocol_counts}")
    logger.info(f"Status counts: {status_counts}")
    
    return protocol_counts, status_counts


class TestConcurrentCommandProcessing(unittest.IsolatedAsyncioTestCase):
    """Test concurrent command processing."""
    
    async def asyncSetUp(self):
        """Set up test environment."""
        # Create mock protocols
        self.hotstuff = MockProtocol("hotstuff", fail_rate=0.1)
        self.raft = MockProtocol("raft", fail_rate=0.2)
        self.cbft = MockProtocol("cbft", fail_rate=0.3)
        
        # Create test configuration
        self.config = {
            "byzantine_command_types": ["financial", "critical", "security"],
            "performance_command_types": ["query", "read", "status"],
            "cluster_command_types": ["local", "cluster", "internal"],
            "distributed_command_types": ["global", "cross_cluster", "network_wide"],
            "security_level_threshold": 7,
            "circuit_breaker": {
                "hotstuff_threshold": 5,
                "raft_threshold": 3,
                "cbft_threshold": 4
            },
            "cache": {
                "max_size": 100,
                "ttl": 60.0
            },
            "adaptive_selection": {
                "enabled": True
            },
            "metrics": {
                "enabled": False
            }
        }
        
        # Create protocol selector
        self.selector = EnhancedConsensusProtocolSelector(
            hotstuff=self.hotstuff,
            raft=self.raft,
            cbft=self.cbft,
            config=self.config
        )
        
        # Start selector
        await self.selector.start()
    
    async def asyncTearDown(self):
        """Clean up test environment."""
        await self.selector.stop()
    
    async def test_concurrent_processing(self):
        """Test concurrent command processing."""
        # Process 50 commands concurrently
        protocol_counts, status_counts = await run_concurrent_tests(self.selector, 50)
        
        # Verify results
        self.assertGreater(protocol_counts["hotstuff"], 0)
        self.assertGreater(protocol_counts["raft"], 0)
        self.assertGreater(protocol_counts["cbft"], 0)
        
        # Total success + error should equal total commands
        self.assertEqual(status_counts["success"] + status_counts["error"], 50)


if __name__ == "__main__":
    # Run basic tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    # Run additional concurrent test
    print("\nRunning additional concurrent command processing test...")
    
    async def run_additional_test():
        # Create test environment
        hotstuff = MockProtocol("hotstuff", fail_rate=0.05)
        raft = MockProtocol("raft", fail_rate=0.1)
        cbft = MockProtocol("cbft", fail_rate=0.15)
        
        # Create protocol selector with all protocols
        selector = EnhancedConsensusProtocolSelector(
            hotstuff=hotstuff,
            raft=raft,
            cbft=cbft
        )
        
        await selector.start()
        
        try:
            # Run concurrent test with 100 commands
            await run_concurrent_tests(selector, 100)
        finally:
            await selector.stop()
    
    # Run additional test
    asyncio.run(run_additional_test())
