#!/usr/bin/env python3
"""
Test Script for Knowledge Graph Optimized Implementation

This script tests the optimized knowledge graph and LLM interface
implementations to verify they work as expected.
"""

import os
import sys
import time
import uuid
import logging
import threading
import json
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("DARF.KG_TEST")

def test_knowledge_graph():
    """Test the knowledge graph implementation."""
    logger.info("Testing Knowledge Graph implementation...")
    
    # Import the knowledge graph
    try:
        from src.modules.knowledge_graph_engine import KnowledgeGraph, Fact
        logger.info("Successfully imported KnowledgeGraph")
    except ImportError as e:
        logger.error(f"Error importing KnowledgeGraph: {e}")
        return False
    
    # Create a temporary file for persistence testing
    temp_file = f"test_kg_{uuid.uuid4()}.json"
    
    try:
        # Create a knowledge graph instance
        kg = KnowledgeGraph(persistence_file=temp_file)
        logger.info(f"Created KG instance: {kg.summary()}")
        
        # Add facts
        facts = []
        for i in range(50):
            fact = Fact(f"entity{i % 10}", f"relates_to", f"entity{(i+5) % 10}", 0.8 + (i % 10) / 100)
            facts.append(fact)
        
        # Add facts in bulk
        for fact in facts:
            kg.add_fact(fact)
        
        logger.info(f"Added {len(facts)} facts: {kg.summary()}")
        
        # Test queries
        results = kg.get_facts(subject="entity1")
        logger.info(f"Query for subject=entity1 returned {len(results)} facts")
        
        results = kg.get_facts(predicate="relates_to")
        logger.info(f"Query for predicate=relates_to returned {len(results)} facts")
        
        results = kg.get_facts(object_val="entity5")
        logger.info(f"Query for object=entity5 returned {len(results)} facts")
        
        # Test pagination
        page = kg.paginate(page_size=10, page=0)
        logger.info(f"Pagination page 0: {len(page['facts'])} facts, total pages: {page['pagination']['total_pages']}")
        
        # Test inference
        if kg.enable_inference:
            kg.add_fact(Fact("entityA", "contains", "entityB"))
            kg.add_fact(Fact("entityB", "contains", "entityC"))
            
            before_count = len(kg.facts)
            inferred = kg.infer_facts()
            after_count = len(kg.facts)
            
            logger.info(f"Inference added {inferred} facts ({before_count} -> {after_count})")
        
        # Test thread safety with concurrent operations
        def concurrent_add_remove():
            """Add and remove facts concurrently."""
            # Add a fact
            fact = Fact(f"thread_{threading.current_thread().name}", "concurrent_op", str(uuid.uuid4()))
            added = kg.add_fact(fact)
            
            # Query something
            kg.get_facts(subject=f"thread_{threading.current_thread().name}")
            
            # Remove the fact
            kg.remove_fact(added.id)
            
            return added.id
        
        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(concurrent_add_remove) for _ in range(100)]
            results = [future.result() for future in futures]
            
        logger.info(f"Completed {len(results)} concurrent operations")
        
        # Test persistence
        logger.info(f"Final KG state: {kg.summary()}")
        
        # Create a new KG instance from the persistence file
        kg2 = KnowledgeGraph(persistence_file=temp_file)
        logger.info(f"Loaded KG from file: {kg2.summary()}")
        
        # Compare the two instances
        logger.info(f"Original KG facts: {len(kg.facts)}, Loaded KG facts: {len(kg2.facts)}")
        
        logger.info("Knowledge Graph test completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error testing Knowledge Graph: {e}")
        return False
    finally:
        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)

def test_llm_interface():
    """Test the LLM interface implementation."""
    logger.info("Testing LLM Graph Interface implementation...")
    
    try:
        # Import the knowledge graph and LLM interface
        from src.modules.knowledge_graph_engine import KnowledgeGraph, Fact
        from src.modules.llm_integration import LLMGraphInterface
        
        logger.info("Successfully imported KnowledgeGraph and LLMGraphInterface")
        
        # Create a knowledge graph instance
        kg = KnowledgeGraph()
        
        # Add some sample facts
        sample_facts = [
            Fact("DARF", "implements", "KnowledgeGraph", 0.9),
            Fact("KnowledgeGraph", "stores", "Facts", 0.95),
            Fact("Facts", "have", "Confidence", 0.8),
            Fact("DARF", "uses", "EventBus", 0.85),
            Fact("EventBus", "enables", "Communication", 0.9),
            Fact("Communication", "improves", "System", 0.75),
            Fact("System", "contains", "Components", 0.9),
            Fact("Components", "include", "KnowledgeGraph", 0.85),
            Fact("Components", "include", "EventBus", 0.85),
            Fact("Components", "include", "LLMInterface", 0.85),
        ]
        
        for fact in sample_facts:
            kg.add_fact(fact)
        
        logger.info(f"Added {len(sample_facts)} sample facts: {kg.summary()}")
        
        # Create the LLM interface
        llm = LLMGraphInterface(kg=kg, config={"cache_enabled": True})
        logger.info("Created LLM interface instance")
        
        # Test querying the knowledge graph
        query = "What does DARF implement?"
        results = llm.query_kg(query)
        logger.info(f"Query '{query}' returned {len(results)} results")
        
        # Test cache
        start_time = time.time()
        results = llm.query_kg(query)  # Should be a cache hit
        end_time = time.time()
        logger.info(f"Cached query took {(end_time - start_time) * 1000:.2f}ms")
        
        # Test entity context
        context = llm.get_entity_context("DARF")
        logger.info(f"Entity context for 'DARF' contains {len(context.get('subject_facts', []))} subject facts and {len(context.get('object_facts', []))} object facts")
        
        # Test adding a fact through the interface
        if hasattr(llm, "add_fact"):
            new_fact = llm.add_fact("DARF", "provides", "Security", 0.7)
            logger.info(f"Added new fact: {new_fact}")
        
        # Test visualization
        viz_data = llm.get_visualization_embeddings(max_entities=10)
        logger.info(f"Visualization contains {len(viz_data.get('labels', []))} entities")
        
        # Test metrics
        metrics = llm.get_metrics()
        logger.info(f"Interface metrics: entity_count={metrics.get('entity_count', 0)}, cache_size={metrics.get('cache_size', 0)}")
        
        logger.info("LLM Interface test completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error testing LLM Interface: {e}")
        return False

def run_tests():
    """Run all tests."""
    results = {
        "knowledge_graph": test_knowledge_graph(),
        "llm_interface": test_llm_interface()
    }
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    
    for test_name, success in results.items():
        status = "PASSED" if success else "FAILED"
        print(f"{test_name}: {status}")
    
    print("=" * 50)
    
    # Return overall success
    return all(results.values())

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
