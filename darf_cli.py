#!/usr/bin/env python3
"""
DARF CLI - Command Line Interface for the DARF Framework

This script provides a command-line interface for interacting with the DARF framework,
making it easier to manage and control the system without writing code.

Usage:
  python darf_cli.py [command] [options]

Commands:
  start                   Start the DARF framework
  stop                    Stop the DARF framework
  status                  Show system status
  facts list              List facts in the knowledge graph
  facts add               Add a new fact to the knowledge graph
  facts delete            Delete a fact from the knowledge graph
  vault store             Store a secret in the vault
  vault retrieve          Retrieve a secret from the vault
  nodes list              List consensus nodes
  nodes status            Show status of a specific node
  nodes recover           Attempt to recover a node from fault state
  events publish          Publish an event
  config show             Show current configuration
  config update           Update configuration
  help                    Show help information
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("DARF.CLI")

# Global reference to DARF framework instance
darf_instance = None


async def initialize_darf(config_file=None):
    """Initialize the DARF framework with the given configuration."""
    global darf_instance
    
    # Import DARF Framework
    try:
        from DARF import DARFFramework
    except ImportError:
        logger.error("Failed to import DARF. Make sure the framework is installed correctly.")
        return None
    
    # Load configuration
    config = None
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return None
    
    # Create DARF Framework instance
    darf_instance = DARFFramework(config)
    
    # Start the framework
    logger.info("Starting DARF framework...")
    success = await darf_instance.start()
    
    if not success:
        logger.error("Failed to start DARF framework")
        return None
    
    logger.info("DARF framework started successfully")
    return darf_instance


async def shutdown_darf():
    """Shutdown the DARF framework."""
    global darf_instance
    
    if not darf_instance:
        logger.warning("DARF framework is not running")
        return True
    
    logger.info("Stopping DARF framework...")
    success = await darf_instance.stop()
    
    if success:
        logger.info("DARF framework stopped successfully")
        darf_instance = None
    else:
        logger.error("Failed to stop DARF framework")
    
    return success


async def cmd_start(args):
    """Start the DARF framework."""
    config_file = args.config
    
    # Check if DARF is already running
    global darf_instance
    if darf_instance:
        logger.warning("DARF framework is already running")
        return 0
    
    # Initialize DARF
    darf = await initialize_darf(config_file)
    
    if darf:
        print("DARF framework started successfully")
        return 0
    else:
        print("Failed to start DARF framework")
        return 1


async def cmd_stop(args):
    """Stop the DARF framework."""
    success = await shutdown_darf()
    
    if success:
        print("DARF framework stopped successfully")
        return 0
    else:
        print("Failed to stop DARF framework")
        return 1


async def cmd_status(args):
    """Show system status."""
    global darf_instance
    
    if not darf_instance:
        print("DARF framework is not running")
        return 1
    
    status = darf_instance.get_system_status()
    
    print("\nDARF System Status:")
    print(f"  Running: {status['running']}")
    print(f"  Startup Time: {status['startup_time']}")
    print(f"  Components: {', '.join(status['components'])}")
    
    if 'consensus' in status and status['consensus']:
        consensus = status['consensus']
        print("\nConsensus Engine:")
        print(f"  Total Nodes: {consensus.get('total_nodes', 'N/A')}")
        print(f"  Stability: {consensus.get('stability', 'N/A')}")
        print(f"  Rounds: {consensus.get('rounds', 'N/A')}")
    
    if 'resources' in status and status['resources']:
        resources = status['resources']
        print("\nSystem Resources:")
        if 'cpu_percent' in resources:
            print(f"  CPU Usage: {resources['cpu_percent']}%")
        if 'memory_percent' in resources:
            print(f"  Memory Usage: {resources['memory_percent']}%")
    
    return 0


async def cmd_facts_list(args):
    """List facts in the knowledge graph."""
    global darf_instance
    
    if not darf_instance:
        print("DARF framework is not running")
        return 1
    
    # Query facts based on filters
    facts = darf_instance.query_facts(
        subject=args.subject,
        predicate=args.predicate,
        object=args.object
    )
    
    if not facts:
        print("No facts found matching the criteria")
        return 0
    
    print(f"\nFound {len(facts)} facts:")
    for i, fact in enumerate(facts):
        print(f"\nFact {i+1}:")
        print(f"  ID: {fact.id}")
        print(f"  Subject: {fact.subject}")
        print(f"  Predicate: {fact.predicate}")
        print(f"  Object: {fact.object}")
        print(f"  Confidence: {fact.confidence}")
        print(f"  Status: {fact.status}")
        if fact.metadata:
            print(f"  Metadata: {json.dumps(fact.metadata, indent=2)}")
    
    return 0


async def cmd_facts_add(args):
    """Add a new fact to the knowledge graph."""
    global darf_instance
    
    if not darf_instance:
        print("DARF framework is not running")
        return 1
    
    # Parse metadata
    metadata = {}
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError:
            print("Error: Metadata must be a valid JSON string")
            return 1
    
    # Add fact
    try:
        fact = await darf_instance.add_fact(
            subject=args.subject,
            predicate=args.predicate,
            object=args.object,
            confidence=args.confidence,
            metadata=metadata
        )
        
        print(f"Fact added successfully with ID: {fact.id}")
        return 0
    except Exception as e:
        print(f"Error adding fact: {e}")
        return 1


async def cmd_facts_delete(args):
    """Delete a fact from the knowledge graph."""
    global darf_instance
    
    if not darf_instance:
        print("DARF framework is not running")
        return 1
    
    # Get knowledge graph component
    kg = darf_instance.get_component("knowledge_graph")
    if not kg:
        print("Knowledge Graph component is not available")
        return 1
    
    # Delete fact
    try:
        success = await kg.remove_fact(args.id)
        
        if success:
            print(f"Fact with ID {args.id} deleted successfully")
            return 0
        else:
            print(f"Fact with ID {args.id} not found or could not be deleted")
            return 1
    except Exception as e:
        print(f"Error deleting fact: {e}")
        return 1


async def cmd_vault_store(args):
    """Store a secret in the vault."""
    global darf_instance
    
    if not darf_instance:
        print("DARF framework is not running")
        return 1
    
    # Read secret data
    secret_data = None
    if args.file:
        try:
            with open(args.file, 'rb') as f:
                secret_data = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return 1
    else:
        secret_data = args.value.encode('utf-8')
    
    # Store secret
    try:
        secret_info = await darf_instance.store_secret(args.path, secret_data)
        
        print(f"Secret stored successfully at path: {args.path}")
        print(f"Secret ID: {secret_info['id']}")
        print(f"Shares required: {secret_info['shares_required']}")
        print(f"Total shares: {secret_info['total_shares']}")
        return 0
    except Exception as e:
        print(f"Error storing secret: {e}")
        return 1


async def cmd_vault_retrieve(args):
    """Retrieve a secret from the vault."""
    global darf_instance
    
    if not darf_instance:
        print("DARF framework is not running")
        return 1
    
    # Get vault component
    vault = darf_instance.get_component("vault")
    if not vault:
        print("Vault component is not available")
        return 1
    
    # Get shares
    shares = []
    if args.shares_dir:
        # Assuming shares are stored as files in the shares directory
        try:
            for file in os.listdir(args.shares_dir):
                if file.startswith(args.id) and '.share.' in file:
                    with open(os.path.join(args.shares_dir, file), 'rb') as f:
                        shares.append(f.read())
        except Exception as e:
            print(f"Error reading shares: {e}")
            return 1
    else:
        # For CLI demo, get shares from local storage
        try:
            for i in range(1, vault.node_count + 1):
                share_path = os.path.join(
                    vault.storage_dir,
                    f"{args.id}.share.{i}"
                )
                if os.path.exists(share_path):
                    with open(share_path, 'rb') as f:
                        shares.append(f.read())
        except Exception as e:
            print(f"Error reading shares: {e}")
            return 1
    
    if len(shares) < vault.threshold:
        print(f"Not enough shares: have {len(shares)}, need {vault.threshold}")
        return 1
    
    # Retrieve secret
    try:
        secret_data = await darf_instance.retrieve_secret(args.path, shares[:vault.threshold])
        
        if not secret_data:
            print(f"Failed to retrieve secret at path: {args.path}")
            return 1
        
        # Output secret
        if args.output:
            with open(args.output, 'wb') as f:
                f.write(secret_data)
            print(f"Secret written to file: {args.output}")
        else:
            try:
                # Try to decode as UTF-8 for display
                secret_str = secret_data.decode('utf-8')
                print(f"Secret value: {secret_str}")
            except UnicodeDecodeError:
                print(f"Secret retrieved (binary data, {len(secret_data)} bytes)")
        
        return 0
    except Exception as e:
        print(f"Error retrieving secret: {e}")
        return 1


async def cmd_nodes_list(args):
    """List consensus nodes."""
    global darf_instance
    
    if not darf_instance:
        print("DARF framework is not running")
        return 1
    
    # Get consensus engine component
    consensus = darf_instance.get_component("consensus_engine")
    if not consensus:
        print("Consensus Engine component is not available")
        return 1
    
    # Get nodes
    nodes = list(consensus.nodes.values())
    
    print(f"\nConsensus Nodes ({len(nodes)} total):")
    for i, node in enumerate(nodes):
        node_info = node.get_state_info()
        print(f"\nNode {i+1}:")
        print(f"  ID: {node_info['id']}")
        print(f"  State: {node_info.get('state', 'Unknown')}")
        print(f"  Last Updated: {node_info.get('last_updated', 'Unknown')}")
        print(f"  Connections: {node_info.get('connections', 0)}")
    
    return 0


async def cmd_nodes_status(args):
    """Show status of a specific node."""
    global darf_instance
    
    if not darf_instance:
        print("DARF framework is not running")
        return 1
    
    # Get consensus engine component
    consensus = darf_instance.get_component("consensus_engine")
    if not consensus:
        print("Consensus Engine component is not available")
        return 1
    
    # Get node
    if args.id not in consensus.nodes:
        print(f"Node with ID {args.id} not found")
        return 1
    
    node = consensus.nodes[args.id]
    node_info = node.get_state_info()
    
    print(f"\nNode Status for {args.id}:")
    for key, value in node_info.items():
        print(f"  {key}: {value}")
    
    return 0


async def cmd_nodes_recover(args):
    """Attempt to recover a node from fault state."""
    global darf_instance
    
    if not darf_instance:
        print("DARF framework is not running")
        return 1
    
    # Get consensus engine component
    consensus = darf_instance.get_component("consensus_engine")
    if not consensus:
        print("Consensus Engine component is not available")
        return 1
    
    # Get node
    if args.id not in consensus.nodes:
        print(f"Node with ID {args.id} not found")
        return 1
    
    node = consensus.nodes[args.id]
    
    # Check if node is in fault state
    node_info = node.get_state_info()
    if node_info.get('state') != "fault_detected":
        print(f"Node {args.id} is not in fault state (current state: {node_info.get('state')})")
        return 1
    
    # Recover node
    try:
        await node.recover()
        print(f"Recovery initiated for node {args.id}")
        return 0
    except Exception as e:
        print(f"Error recovering node: {e}")
        return 1


async def cmd_events_publish(args):
    """Publish an event."""
    global darf_instance
    
    if not darf_instance:
        print("DARF framework is not running")
        return 1
    
    # Parse event data
    event_data = {}
    if args.data:
        try:
            event_data = json.loads(args.data)
        except json.JSONDecodeError:
            print("Error: Event data must be a valid JSON string")
            return 1
    
    # Publish event
    try:
        from src.modules.event_bus_telemetry.event_bus_telemetry_fixed import Event
        event_bus = darf_instance.get_component("event_bus")
        
        if not event_bus:
            print("Event Bus component is not available")
            return 1
        
        await event_bus.publish(Event(
            event_type=args.type,
            source=args.source,
            data=event_data
        ))
        
        print(f"Event published successfully: {args.type}")
        return 0
    except Exception as e:
        print(f"Error publishing event: {e}")
        return 1


async def cmd_config_show(args):
    """Show current configuration."""
    global darf_instance
    
    if not darf_instance:
        print("DARF framework is not running")
        return 1
    
    print("\nCurrent DARF Configuration:")
    print(json.dumps(darf_instance.config, indent=2))
    return 0


async def cmd_config_update(args):
    """Update configuration."""
    global darf_instance
    
    if not darf_instance:
        print("DARF framework is not running")
        return 1
    
    print("Configuration can only be updated when starting the DARF framework.")
    print("Please stop the framework and restart it with the updated configuration.")
    return 1


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="DARF Command Line Interface",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start the DARF framework")
    start_parser.add_argument(
        "--config", 
        type=str, 
        default="darf_config.json",
        help="Path to configuration file"
    )
    
    # Stop command
    subparsers.add_parser("stop", help="Stop the DARF framework")
    
    # Status command
    subparsers.add_parser("status", help="Show system status")
    
    # Facts commands
    facts_parser = subparsers.add_parser("facts", help="Knowledge Graph operations")
    facts_subparsers = facts_parser.add_subparsers(dest="facts_command", help="Facts subcommand")
    
    # Facts list command
    facts_list_parser = facts_subparsers.add_parser("list", help="List facts in the knowledge graph")
    facts_list_parser.add_argument("--subject", type=str, help="Filter by subject")
    facts_list_parser.add_argument("--predicate", type=str, help="Filter by predicate")
    facts_list_parser.add_argument("--object", type=str, help="Filter by object")
    
    # Facts add command
    facts_add_parser = facts_subparsers.add_parser("add", help="Add a new fact to the knowledge graph")
    facts_add_parser.add_argument("--subject", type=str, required=True, help="Subject of the fact")
    facts_add_parser.add_argument("--predicate", type=str, required=True, help="Predicate of the fact")
    facts_add_parser.add_argument("--object", type=str, required=True, help="Object of the fact")
    facts_add_parser.add_argument("--confidence", type=float, default=1.0, help="Confidence level (0-1)")
    facts_add_parser.add_argument("--metadata", type=str, help="Metadata as JSON string")
    
    # Facts delete command
    facts_delete_parser = facts_subparsers.add_parser("delete", help="Delete a fact from the knowledge graph")
    facts_delete_parser.add_argument("--id", type=str, required=True, help="ID of the fact to delete")
    
    # Vault commands
    vault_parser = subparsers.add_parser("vault", help="Vault operations")
    vault_subparsers = vault_parser.add_subparsers(dest="vault_command", help="Vault subcommand")
    
    # Vault store command
    vault_store_parser = vault_subparsers.add_parser("store", help="Store a secret in the vault")
    vault_store_parser.add_argument("--path", type=str, required=True, help="Path for the secret")
    vault_store_group = vault_store_parser.add_mutually_exclusive_group(required=True)
    vault_store_group.add_argument("--value", type=str, help="Secret value as string")
    vault_store_group.add_argument("--file", type=str, help="Path to file containing the secret")
    
    # Vault retrieve command
    vault_retrieve_parser = vault_subparsers.add_parser("retrieve", help="Retrieve a secret from the vault")
    vault_retrieve_parser.add_argument("--path", type=str, required=True, help="Path of the secret")
    vault_retrieve_parser.add_argument("--id", type=str, required=True, help="ID of the secret")
    vault_retrieve_parser.add_argument("--shares-dir", type=str, help="Directory containing share files")
    vault_retrieve_parser.add_argument("--output", type=str, help="Output file for the secret")
    
    # Nodes commands
    nodes_parser = subparsers.add_parser("nodes", help="Consensus nodes operations")
    nodes_subparsers = nodes_parser.add_subparsers(dest="nodes_command", help="Nodes subcommand")
    
    # Nodes list command
    nodes_subparsers.add_parser("list", help="List consensus nodes")
    
    # Nodes status command
    nodes_status_parser = nodes_subparsers.add_parser("status", help="Show status of a specific node")
    nodes_status_parser.add_argument("--id", type=str, required=True, help="ID of the node")
    
    # Nodes recover command
    nodes_recover_parser = nodes_subparsers.add_parser("recover", help="Attempt to recover a node from fault state")
    nodes_recover_parser.add_argument("--id", type=str, required=True, help="ID of the node to recover")
    
    # Events commands
    events_parser = subparsers.add_parser("events", help="Event operations")
    events_subparsers = events_parser.add_subparsers(dest="events_command", help="Events subcommand")
    
    # Events publish command
    events_publish_parser = events_subparsers.add_parser("publish", help="Publish an event")
    events_publish_parser.add_argument("--type", type=str, required=True, help="Event type")
    events_publish_parser.add_argument("--source", type=str, default="cli", help="Event source")
    events_publish_parser.add_argument("--data", type=str, help="Event data as JSON string")
    
    # Config commands
    config_parser = subparsers.add_parser("config", help="Configuration operations")
    config_subparsers = config_parser.add_subparsers(dest="config_command", help="Config subcommand")
    
    # Config show command
    config_subparsers.add_parser("show", help="Show current configuration")
    
    # Config update command
    config_subparsers.add_parser("update", help="Update configuration")
    
    # Help command
    subparsers.add_parser("help", help="Show help information")
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command or args.command == "help":
        parser.print_help()
        return 0
    
    # Map commands to functions
    command_map = {
        "start": cmd_start,
        "stop": cmd_stop,
        "status": cmd_status,
        "facts": {
            "list": cmd_facts_list,
            "add": cmd_facts_add,
            "delete": cmd_facts_delete
        },
        "vault": {
            "store": cmd_vault_store,
            "retrieve": cmd_vault_retrieve
        },
        "nodes": {
            "list": cmd_nodes_list,
            "status": cmd_nodes_status,
            "recover": cmd_nodes_recover
        },
        "events": {
            "publish": cmd_events_publish
        },
        "config": {
            "show": cmd_config_show,
            "update": cmd_config_update
        }
    }
    
    # Execute command
    if args.command in command_map:
        command_func = command_map[args.command]
        
        # Handle subcommands
        if isinstance(command_func, dict):
            subcommand = getattr(args, f"{args.command}_command")
            if not subcommand:
                print(f"Error: Please specify a subcommand for '{args.command}'")
                return 1
                
            if subcommand in command_func:
                command_func = command_func[subcommand]
            else:
                print(f"Error: Unknown subcommand '{subcommand}' for '{args.command}'")
                return 1
        
        # Run command function
        try:
            return asyncio.run(command_func(args))
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return 1
        except Exception as e:
            print(f"Error executing command: {e}")
            return 1
    else:
        print(f"Error: Unknown command '{args.command}'")
        return 1


if __name__ == "__main__":
    sys.exit(main())
