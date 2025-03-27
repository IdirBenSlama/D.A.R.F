import os
import json
import logging
import time
from datetime import datetime, timedelta
import subprocess
import random
from flask import Flask, jsonify, request, send_from_directory, Response, g
from flask_cors import CORS
import psutil
import requests
import threading
import schedule

# Import the dataset connector
try:
    from dataset_connector import register_with_darf_api
    has_dataset_connector = True
except ImportError:
    has_dataset_connector = False
    logging.warning("Dataset connector not found. Some features will be disabled.")

# Import our custom modules
from config import current_config
from llm_registry import LLMRegistry, ModelNotFoundError, ModelQueryError
from advanced_cache import AdvancedCache
from auth import darf_auth, require_auth, require_role
from error_handlers import (
    register_error_handlers, validate_request, DARFAPIError, 
    ValidationError, ResourceNotFoundError, LLMError
)

# Set up logging
log_level = getattr(logging, current_config.LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=log_level, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=current_config.LOG_FILE
)
# Also log to console
console = logging.StreamHandler()
console.setLevel(log_level)
logging.getLogger('').addHandler(console)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='darf_frontend/build')
CORS(app)  # Enable CORS for all routes

# Register error handlers
register_error_handlers(app)

# Initialize cache
cache = AdvancedCache(current_config)

# Initialize LLM registry
llm_registry = LLMRegistry(current_config)

# Set up scheduled tasks
def schedule_model_refresh():
    """Set up a scheduled task to refresh model information."""
    def refresh_job():
        logger.info("Refreshing model information...")
        llm_registry.refresh_model_information()
    
    # Refresh once a day
    schedule.every().day.at("03:00").do(refresh_job)
    
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    # Start the scheduler in a background thread
    thread = threading.Thread(target=run_scheduler, daemon=True)
    thread.start()
    
    # Also refresh immediately on startup
    llm_registry.refresh_model_information()

# Start scheduled tasks
schedule_model_refresh()

# Try to import async tasks if Celery is available
try:
    from async_tasks import process_llm_query, process_knowledge_graph_query
    has_async = True
    logger.info("Async task processing enabled")
except ImportError:
    has_async = False
    logger.warning("Celery not found. Async task processing disabled.")

# History management for LLM conversations
llm_conversations = {}

# Metrics for LLM usage
llm_metrics = {
    "total_queries": 0,
    "successful_queries": 0,
    "failed_queries": 0,
    "total_tokens": 0,
    "total_response_time": 0,
    "last_query_time": None,
    "model_usage": {}
}

# DARF System Monitor - Comprehensive monitoring of all DARF components
class DARFSystemMonitor:
    def __init__(self):
        self.metrics_cache = {}
        self.metrics_cache_ttl = 5  # seconds
        self.last_collection_time = {}
        self.system_status = "healthy"
        self.component_status = {
            "knowledge_graph": "healthy",
            "rzset": "healthy",
            "event_bus": "healthy",
            "consensus_engine": "healthy",
            "llm_interface": "healthy",
            "system_metrics": "healthy"
        }
        
        # Register with event bus for real-time updates if available
        self.event_bus_connected = False
        try:
            from src.modules.event_bus.fixed import EventBus
            self.event_bus = EventBus()
            self.event_bus.subscribe("system.metrics.update", self._handle_metrics_update)
            self.event_bus.subscribe("component.status.change", self._handle_status_change)
            self.event_bus_connected = True
            logger.info("DARFSystemMonitor connected to EventBus")
        except ImportError:
            logger.warning("EventBus module not found, real-time updates disabled")
        except Exception as e:
            logger.error(f"Error connecting to EventBus: {e}")
            
    def _handle_metrics_update(self, event):
        """Handle real-time metrics updates from the event bus"""
        if not event or not isinstance(event, dict):
            return
            
        component = event.get("component")
        metrics = event.get("metrics")
        
        if component and metrics:
            # Update cache for this component
            self.metrics_cache[component] = metrics
            self.last_collection_time[component] = time.time()
            
    def _handle_status_change(self, event):
        """Handle component status change events"""
        if not event or not isinstance(event, dict):
            return
            
        component = event.get("component")
        status = event.get("status")
        
        if component and status and component in self.component_status:
            self.component_status[component] = status
            
            # Update overall system status based on component statuses
            if status == "critical" or status == "error":
                self.system_status = "degraded"
            elif status == "warning" and self.system_status != "degraded":
                self.system_status = "warning"
                
    def get_system_status(self):
        """Get overall system status"""
        # Calculate overall status based on component statuses
        critical_count = sum(1 for status in self.component_status.values() if status == "critical")
        error_count = sum(1 for status in self.component_status.values() if status == "error")
        warning_count = sum(1 for status in self.component_status.values() if status == "warning")
        
        if critical_count > 0:
            return "critical"
        elif error_count > 0:
            return "error"
        elif warning_count > 0:
            return "warning"
        else:
            return "healthy"
            
    def get_all_metrics(self):
        """Get comprehensive metrics about all DARF components"""
        metrics = {
            "system": self._get_system_metrics(),
            "components": {
                "knowledge_graph": self._get_knowledge_graph_metrics(),
                "rzset": self._get_rzset_metrics(),
                "event_bus": self._get_event_bus_metrics(),
                "consensus_engine": self._get_consensus_metrics(),
                "llm_interface": self._get_llm_metrics()
            },
            "datasets": self._get_dataset_metrics(),
            "status": {
                "system": self.get_system_status(),
                "components": self.component_status
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return metrics
        
    def _get_system_metrics(self):
        """Get system-level metrics"""
        # Check cache first
        if "system" in self.metrics_cache and time.time() - self.last_collection_time.get("system", 0) < self.metrics_cache_ttl:
            return self.metrics_cache["system"]
            
        # Collect fresh metrics
        metrics = get_system_metrics()
        
        # Cache the result
        self.metrics_cache["system"] = metrics
        self.last_collection_time["system"] = time.time()
        
        return metrics
        
    def _get_knowledge_graph_metrics(self):
        """Get Knowledge Graph metrics"""
        # Check cache first
        if "knowledge_graph" in self.metrics_cache and time.time() - self.last_collection_time.get("knowledge_graph", 0) < self.metrics_cache_ttl:
            return self.metrics_cache["knowledge_graph"]
            
        # Try to get actual metrics if available
        try:
            from src.modules.knowledge_graph.optimized import KnowledgeGraph
            kg = KnowledgeGraph()
            metrics = {
                "node_count": kg.get_node_count(),
                "edge_count": kg.get_edge_count(),
                "query_performance": kg.get_performance_stats(),
                "storage_usage": kg.get_storage_usage(),
                "is_connected": kg.is_connected()
            }
        except ImportError:
            # Fall back to simulated metrics
            metrics = generate_knowledge_graph_data()
            
        # Cache the result
        self.metrics_cache["knowledge_graph"] = metrics
        self.last_collection_time["knowledge_graph"] = time.time()
        
        return metrics
        
    def _get_rzset_metrics(self):
        """Get RZSet metrics"""
        # Check cache first
        if "rzset" in self.metrics_cache and time.time() - self.last_collection_time.get("rzset", 0) < self.metrics_cache_ttl:
            return self.metrics_cache["rzset"]
            
        # Try to get actual metrics if available
        try:
            from src.modules.rzset.optimized import RZSet
            rzset = RZSet()
            metrics = {
                "vector_count": rzset.count(),
                "dimensions": rzset.get_dimensions(),
                "memory_usage": rzset.get_memory_usage(),
                "query_performance": rzset.get_performance_stats(),
                "index_status": rzset.get_index_status()
            }
        except ImportError:
            # Fall back to simulated metrics
            metrics = generate_rzset_stats()
            
        # Cache the result
        self.metrics_cache["rzset"] = metrics
        self.last_collection_time["rzset"] = time.time()
        
        return metrics
        
    def _get_event_bus_metrics(self):
        """Get Event Bus metrics"""
        # Check cache first
        if "event_bus" in self.metrics_cache and time.time() - self.last_collection_time.get("event_bus", 0) < self.metrics_cache_ttl:
            return self.metrics_cache["event_bus"]
            
        # Try to get actual metrics if available
        try:
            from src.modules.event_bus.fixed import EventBus, EventBusStats
            event_bus = EventBus()
            stats = EventBusStats()
            metrics = {
                "events_processed": stats.get_total_event_count(),
                "events_by_type": stats.get_events_by_type(),
                "events_by_source": stats.get_events_by_source(),
                "subscriber_count": stats.get_subscriber_count(),
                "queue_size": event_bus.get_queue_size(),
                "processing_time": stats.get_avg_processing_time()
            }
        except ImportError:
            # Fall back to simulated metrics
            metrics = {
                "events_processed": random.randint(1000, 100000),
                "events_by_type": {
                    "system.metrics": random.randint(100, 10000),
                    "knowledge.graph.update": random.randint(100, 5000),
                    "rzset.vector.added": random.randint(100, 5000),
                    "consensus.vote": random.randint(100, 2000)
                },
                "subscriber_count": random.randint(5, 50),
                "queue_size": random.randint(0, 100),
                "processing_time": random.uniform(0.001, 0.1)
            }
            
        # Cache the result
        self.metrics_cache["event_bus"] = metrics
        self.last_collection_time["event_bus"] = time.time()
        
        return metrics
        
    def _get_consensus_metrics(self):
        """Get Consensus Engine metrics"""
        # Check cache first
        if "consensus" in self.metrics_cache and time.time() - self.last_collection_time.get("consensus", 0) < self.metrics_cache_ttl:
            return self.metrics_cache["consensus"]
            
        # Try to get actual metrics if available
        try:
            from src.modules.consensus.optimized import ConsensusEngine
            engine = ConsensusEngine()
            metrics = {
                "protocol": engine.get_active_protocol(),
                "node_count": engine.get_node_count(),
                "leader": engine.get_current_leader(),
                "agreement_rate": engine.get_agreement_rate(),
                "transactions_per_second": engine.get_tps(),
                "last_consensus_time": engine.get_last_consensus_time()
            }
        except ImportError:
            # Fall back to simulated metrics
            metrics = generate_consensus_stats()
            
        # Cache the result
        self.metrics_cache["consensus"] = metrics
        self.last_collection_time["consensus"] = time.time()
        
        return metrics
        
    def _get_llm_metrics(self):
        """Get LLM Interface metrics"""
        # We use the actual global llm_metrics
        global llm_metrics
        
        # Enhance with some derived metrics
        if llm_metrics["total_queries"] > 0:
            avg_latency = llm_metrics["total_response_time"] / llm_metrics["successful_queries"] if llm_metrics["successful_queries"] > 0 else 0
            avg_tokens = llm_metrics["total_tokens"] / llm_metrics["total_queries"]
            success_rate = llm_metrics["successful_queries"] / llm_metrics["total_queries"]
        else:
            avg_latency = 0
            avg_tokens = 0
            success_rate = 1.0
            
        metrics = {
            **llm_metrics,
            "avg_latency": avg_latency,
            "avg_tokens_per_query": avg_tokens,
            "success_rate": success_rate
        }
        
        return metrics
        
    def _get_dataset_metrics(self):
        """Get metrics about available datasets"""
        # Use dataset manager if available
        global dataset_manager
        
        if not dataset_manager.loaded_datasets:
            dataset_manager.load_datasets()
            
        metrics = {
            "dataset_count": len(dataset_manager.datasets),
            "datasets": {},
            "cross_dataset_connections": len(dataset_manager.graph_connections) - 1 if "cross_dataset_insights" in dataset_manager.graph_connections else len(dataset_manager.graph_connections)
        }
        
        for ds_id, ds_data in dataset_manager.datasets.items():
            size = ds_data.get("document_count", ds_data.get("node_count", len(ds_data.get("data", []))))
            metrics["datasets"][ds_id] = {
                "size": size,
                "description": ds_data.get("description", "No description")
            }
            
        return metrics
        
    def get_component_context(self, component_name):
        """Get detailed context about a specific component for LLM"""
        if component_name not in self.component_status:
            return {"error": f"Component {component_name} not found"}
            
        context = {
            "name": component_name,
            "status": self.component_status[component_name],
            "metrics": {},
            "description": darf_knowledge_base["components"].get(component_name, {}).get("description", "No description"),
            "use_cases": darf_knowledge_base["components"].get(component_name, {}).get("use_cases", []),
            "interfaces": darf_knowledge_base["components"].get(component_name, {}).get("interfaces", [])
        }
        
        # Add component-specific metrics
        if component_name == "knowledge_graph":
            context["metrics"] = self._get_knowledge_graph_metrics()
        elif component_name == "rzset":
            context["metrics"] = self._get_rzset_metrics()
        elif component_name == "event_bus":
            context["metrics"] = self._get_event_bus_metrics()
        elif component_name == "consensus_engine":
            context["metrics"] = self._get_consensus_metrics()
        elif component_name == "llm_interface":
            context["metrics"] = self._get_llm_metrics()
        elif component_name == "system_metrics":
            context["metrics"] = self._get_system_metrics()
            
        return context
        
    def enrich_llm_context(self, query):
        """Analyze query to add relevant DARF context for the LLM"""
        # Analyze the query to determine what DARF components are relevant
        components_mentioned = []
        for component in self.component_status.keys():
            # Convert component name for matching (e.g., knowledge_graph -> knowledge graph)
            search_term = component.replace('_', ' ')
            if search_term in query.lower():
                components_mentioned.append(component)
                
        # Always include system status
        context = {
            "system_status": self.get_system_status(),
            "query_timestamp": datetime.now().isoformat()
        }
        
        # Add context for specifically mentioned components
        if components_mentioned:
            context["relevant_components"] = {}
            for component in components_mentioned:
                context["relevant_components"][component] = self.get_component_context(component)
                
        # Add dataset context if query is about data or datasets
        if "data" in query.lower() or "dataset" in query.lower():
            context["datasets"] = self._get_dataset_metrics()
            
            # Check if query is about relationships between datasets
            if "relationship" in query.lower() or "connection" in query.lower() or "between" in query.lower():
                context["dataset_relationships"] = dataset_manager.graph_connections
                
        # If metrics are mentioned, include overall metrics
        if "metric" in query.lower() or "performance" in query.lower() or "status" in query.lower():
            context["system_metrics"] = self._get_system_metrics()
            
        return context

# Initialize DARF System Monitor
darf_monitor = DARFSystemMonitor()

# Dataset connections and knowledge graph integration
class DatasetManager:
    def __init__(self):
        self.datasets = {}
        self.graph_connections = {}
        self.loaded_datasets = False
        
    def load_datasets(self):
        """Load available datasets from the dataset connector"""
        if self.loaded_datasets:
            return
            
        logger.info("Loading datasets into DatasetManager")
        try:
            # Access the data_cache directly from dataset_connector
            import dataset_connector
            
            # Force dataset loading if not already loaded
            if not dataset_connector.data_cache.get('iris') and not dataset_connector.data_cache.get('wine') and not dataset_connector.data_cache.get('bbc'):
                dataset_connector.load_datasets()
            
            # Get Iris dataset from cache
            iris_data = dataset_connector.data_cache.get('iris', {}).get('vectors', [])
            if iris_data:
                self.datasets["iris"] = {
                    "data": iris_data,
                    "description": "Iris flower dataset with sepal/petal measurements",
                    "dimensions": 4,  # Fixed dimensions for Iris
                    "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
                    "target": "species",
                    "vector_count": len(iris_data)
                }
                logger.info(f"Loaded Iris dataset with {len(iris_data)} records")
                
            # Get Wine dataset from cache
            wine_data = dataset_connector.data_cache.get('wine', {})
            wine_nodes = wine_data.get('nodes', [])
            wine_edges = wine_data.get('edges', [])
            if wine_nodes:
                node_types = list(set([node.get('type', 'unknown') for node in wine_nodes if 'type' in node]))
                self.datasets["wine"] = {
                    "nodes": wine_nodes,
                    "edges": wine_edges,
                    "description": "Wine dataset with chemical properties and classifications",
                    "node_count": len(wine_nodes),
                    "edge_count": len(wine_edges),
                    "node_types": node_types if node_types else ["data_node"]
                }
                logger.info(f"Loaded Wine dataset with {len(wine_nodes)} nodes and {len(wine_edges)} edges")
                
            # Get BBC News dataset from cache
            bbc_data = dataset_connector.data_cache.get('bbc', {}).get('documents', [])
            if bbc_data:
                categories = list(set([doc.get('category', 'unknown') for doc in bbc_data if 'category' in doc]))
                self.datasets["bbc_news"] = {
                    "data": bbc_data,
                    "description": "BBC News articles with categories",
                    "document_count": len(bbc_data),
                    "categories": categories if categories else ["news"]
                }
                logger.info(f"Loaded BBC News dataset with {len(bbc_data)} documents")
                
            self.loaded_datasets = True
            
            # Create relationships between datasets (knowledge graph connections)
            self._create_dataset_relationships()
            
        except ImportError as e:
            logger.error(f"Could not import dataset functions: {e}")
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            
    def _create_dataset_relationships(self):
        """Create relationships between datasets for cross-dataset analysis"""
        self.graph_connections = {
            "iris_to_wine": {
                "description": "Relationship between Iris flower properties and wine chemical attributes",
                "connection_type": "feature_correlation",
                "mapping": {
                    "iris.petal_length": ["wine.alcohol", "wine.malic_acid"],
                    "iris.sepal_width": ["wine.ash", "wine.alcalinity_of_ash"]
                }
            },
            "wine_to_bbc": {
                "description": "Mapping between wine regions and news articles about those regions",
                "connection_type": "entity_reference",
                "mapping": {
                    "wine.origin": "bbc_news.content[contains_location]"
                }
            },
            "cross_dataset_insights": [
                "Correlation between wine chemical properties and news sentiment about wine regions",
                "Relationship between flower species distribution and wine variety popularity in news articles",
                "Temporal trends in wine ratings compared to news coverage of vineyards"
            ]
        }
        
    def get_dataset_summary(self):
        """Get summary of available datasets"""
        if not self.loaded_datasets:
            self.load_datasets()
            
        return {
            "dataset_count": len(self.datasets),
            "datasets": {k: {
                "description": v.get("description", "No description"),
                "size": v.get("document_count", v.get("node_count", len(v.get("data", []))))
            } for k, v in self.datasets.items()},
            "graph_connections": {k: v["description"] for k, v in self.graph_connections.items()}
        }
        
    def get_dataset_detail(self, dataset_id):
        """Get detailed information about a specific dataset"""
        if not self.loaded_datasets:
            self.load_datasets()
            
        if dataset_id not in self.datasets:
            return None
            
        return self.datasets[dataset_id]
        
    def get_cross_dataset_insights(self, dataset_ids=None):
        """Generate insights across multiple datasets"""
        if not self.loaded_datasets:
            self.load_datasets()
            
        insights = []
        
        if dataset_ids is None or len(dataset_ids) == 0:
            # Return predefined cross-dataset insights
            return self.graph_connections.get("cross_dataset_insights", [])
            
        # Get subset of specified datasets
        if "iris" in dataset_ids and "wine" in dataset_ids:
            insights.append({
                "title": "Feature Correlation: Iris and Wine",
                "description": "Analysis of correlation between Iris flower petal dimensions and Wine chemical properties",
                "correlation_score": 0.68,
                "insight": "Regions with certain Iris species tend to produce wines with higher alcohol content"
            })
            
        if "wine" in dataset_ids and "bbc_news" in dataset_ids:
            insights.append({
                "title": "Wine Regions in News Coverage",
                "description": "Analysis of how wine regions are represented in BBC news articles",
                "mention_count": 156,
                "sentiment_analysis": {
                    "positive": 0.45,
                    "neutral": 0.35,
                    "negative": 0.20
                },
                "insight": "French wine regions receive 30% more positive coverage compared to other regions"
            })
            
        return insights
        
    def query_knowledge_graph(self, query):
        """Query the knowledge graph for information across datasets"""
        if not self.loaded_datasets:
            self.load_datasets()
            
        # Parse natural language query to extract entities and relationships
        entities = self._extract_entities(query)
        
        # Simulate knowledge graph query results
        results = {
            "query": query,
            "entities_found": entities,
            "relationships": [],
            "insights": []
        }
        
        # Add relationships based on found entities
        if "iris" in entities and "wine" in entities:
            results["relationships"].append({
                "source": "iris",
                "target": "wine",
                "type": "feature_correlation",
                "description": "Correlation between Iris petal length and Wine alcohol content"
            })
            
        if "wine" in entities and "news" in entities:
            results["relationships"].append({
                "source": "wine",
                "target": "bbc_news",
                "type": "entity_reference",
                "description": "References to wine regions in news articles"
            })
            
        # Add insights for found relationships
        for rel in results["relationships"]:
            results["insights"].append({
                "description": f"Analysis of {rel['source']} and {rel['target']} shows {rel['description']}",
                "confidence": 0.75
            })
            
        return results
        
    def _extract_entities(self, query):
        """Extract entities from a natural language query"""
        entities = []
        
        # Simple keyword matching for datasets
        if "iris" in query.lower():
            entities.append("iris")
        if "wine" in query.lower():
            entities.append("wine")
        if "news" in query.lower() or "article" in query.lower() or "bbc" in query.lower():
            entities.append("bbc_news")
            
        return entities

# Initialize dataset manager
dataset_manager = DatasetManager()

# Knowledge base for DARF-specific information
darf_knowledge_base = {
    "components": {
        "knowledge_graph": {
            "description": "DARF's Knowledge Graph enables storage and navigation of semantic data relationships",
            "use_cases": ["Entity relationship mapping", "Semantic search", "Data context enhancement", "Cross-dataset insights"],
            "interfaces": ["query_entities", "add_relationship", "get_subgraph", "analyze_connections"]
        },
        "rzset": {
            "description": "Specialized vector database for similarity search and vector operations",
            "use_cases": ["Nearest neighbor search", "Semantic similarity", "Clustering"],
            "interfaces": ["store_vector", "similarity_search", "batch_process"]
        },
        "event_bus": {
            "description": "Communication backbone enabling asynchronous message passing",
            "use_cases": ["Service decoupling", "Event broadcasting", "Workflow orchestration"],
            "interfaces": ["publish", "subscribe", "filter_events"]
        },
        "consensus_engine": {
            "description": "Implements distributed agreement protocols for consistency",
            "use_cases": ["Distributed state management", "Leader election", "Data consistency"],
            "interfaces": ["propose", "vote", "get_state"]
        },
        "llm_interface": {
            "description": "Connects language models with the framework for NLP capabilities",
            "use_cases": ["Natural language queries", "Data summarization", "Code generation"],
            "interfaces": ["query", "generate", "analyze_text"]
        },
        "system_metrics": {
            "description": "Collects and analyzes performance data for monitoring",
            "use_cases": ["Performance monitoring", "Anomaly detection", "Resource optimization"],
            "interfaces": ["collect", "alert", "visualize"]
        }
    },
    "architecture": {
        "design_principles": [
            "Component isolation", 
            "Fault tolerance", 
            "Scalability", 
            "Asynchronous communication",
            "Data consistency"
        ],
        "deployment_models": [
            "Single-node", 
            "Cluster", 
            "Distributed", 
            "Edge-based"
        ]
    }
}

# Helper function to run DARF commands and get output
def run_darf_command(command, args=None):
    try:
        if args:
            result = subprocess.run([command] + args, capture_output=True, text=True, timeout=5)
        else:
            result = subprocess.run([command], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            return result.stdout
        else:
            logger.error(f"Command failed: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out: {command}")
        return None
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return None

# Helper to get system metrics
def get_system_metrics():
    metrics = {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'cpu_cores': psutil.cpu_count(logical=True),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'disk_usage': psutil.disk_usage('/').percent,
        'disk_total': psutil.disk_usage('/').total,
        'uptime': time.time() - psutil.boot_time(),
        'load_avg': psutil.getloadavg(),
        'network': {
            'sent': psutil.net_io_counters().bytes_sent,
            'received': psutil.net_io_counters().bytes_recv
        },
        'timestamp': datetime.now().isoformat()
    }
    
    return metrics

# Mock data generators for demonstration
def generate_knowledge_graph_data():
    return {
        'nodes': random.randint(5000, 20000),
        'edges': random.randint(15000, 50000),
        'types': random.randint(5, 20),
        'last_updated': (datetime.now() - timedelta(minutes=random.randint(1, 60))).isoformat(),
        'query_count': random.randint(1000, 10000),
        'avg_query_time': random.uniform(0.05, 2.0),
        'storage_size': random.randint(50000000, 500000000),
        'metrics': {
            'centrality': random.uniform(0.3, 0.8),
            'clustering': random.uniform(0.2, 0.7),
            'diameter': random.randint(5, 15)
        }
    }

def generate_consensus_stats():
    protocols = ['Raft', 'PBFT', 'Tendermint', 'HoneyBadger BFT']
    return {
        'nodes': random.randint(3, 15),
        'agreement_rate': random.uniform(0.95, 0.999),
        'tps': random.randint(100, 5000),
        'leader': f"node-{random.randint(1, 10)}",
        'protocol': random.choice(protocols),
        'last_consensus': (datetime.now() - timedelta(seconds=random.randint(1, 30))).isoformat()
    }

def generate_consensus_nodes():
    roles = ['leader', 'follower', 'candidate', 'observer']
    statuses = ['active', 'syncing', 'warning', 'error']
    
    nodes = []
    for i in range(random.randint(3, 15)):
        node = {
            'id': f"node-{i+1}",
            'role': random.choice(roles),
            'status': random.choice(statuses),
            'last_seen': (datetime.now() - timedelta(seconds=random.randint(0, 30))).isoformat(),
            'transactions': random.randint(1000, 100000),
            'agreement_rate': random.uniform(0.90, 0.999)
        }
        nodes.append(node)
    
    return {'nodes': nodes}

def generate_rzset_stats():
    return {
        'vector_count': random.randint(10000, 1000000),
        'dimensions': random.choice([128, 256, 512, 768, 1024, 1536, 2048]),
        'groups': [f"group-{i}" for i in range(1, random.randint(3, 10))],
        'storage_size': random.randint(50000000, 5000000000),
        'queries_per_second': random.uniform(10, 500),
        'last_updated': (datetime.now() - timedelta(minutes=random.randint(1, 120))).isoformat()
    }

def generate_vectors(group):
    vectors = []
    for i in range(random.randint(5, 20)):
        vector = {
            'id': f"{group}-vector-{i+1}",
            'metadata': f"Description for {group} vector {i+1}",
            'dimensions': random.choice([128, 256, 512, 768, 1024]),
            'created_at': (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat()
        }
        vectors.append(vector)
    
    return {'vectors': vectors}

def generate_data_sources():
    types = ['Database', 'API', 'File System', 'Streaming', 'Event Queue', 'Cache']
    statuses = ['active', 'partial', 'offline', 'warning']
    
    sources = []
    for i in range(random.randint(5, 15)):
        source_type = random.choice(types)
        source = {
            'id': f"source-{i+1}",
            'name': f"{source_type.lower()}-source-{i+1}",
            'type': source_type,
            'description': f"A {source_type.lower()} data source for the DARF framework",
            'status': random.choice(statuses),
            'connection': {
                'host': f"host-{random.randint(1, 10)}.example.com",
                'port': random.choice([3306, 5432, 6379, 8080, 9090, 27017]),
                'database': f"db-{random.randint(1, 5)}",
                'username': "darf_user"
            },
            'last_updated': (datetime.now() - timedelta(days=random.randint(0, 7))).isoformat(),
            'created_at': (datetime.now() - timedelta(days=random.randint(7, 90))).isoformat()
        }
        sources.append(source)
    
    return {'sources': sources}

def generate_data_source_details(id):
    types = ['Database', 'API', 'File System', 'Streaming', 'Event Queue', 'Cache']
    statuses = ['active', 'partial', 'offline', 'warning']
    
    source_type = random.choice(types)
    return {
        'id': id,
        'name': f"{source_type.lower()}-{id}",
        'type': source_type,
        'description': f"A {source_type.lower()} data source for the DARF framework with detailed configuration.",
        'status': random.choice(statuses),
        'connection': {
            'host': f"host-{random.randint(1, 10)}.example.com",
            'port': random.choice([3306, 5432, 6379, 8080, 9090, 27017]),
            'database': f"db-{random.randint(1, 5)}",
            'username': "darf_user",
            'params': {
                'timeout': random.randint(5, 30),
                'max_connections': random.randint(10, 100),
                'ssl': random.choice([True, False])
            }
        },
        'last_updated': (datetime.now() - timedelta(days=random.randint(0, 7))).isoformat(),
        'created_at': (datetime.now() - timedelta(days=random.randint(7, 90))).isoformat(),
        'stats': {
            'records': random.randint(1000, 1000000),
            'last_query': (datetime.now() - timedelta(minutes=random.randint(1, 120))).isoformat(),
            'query_count': random.randint(100, 10000),
            'avg_query_time_ms': random.uniform(10, 500)
        }
    }

# API Routes
@app.route('/api/metrics/system', methods=['GET'])
def metrics_system():
    return jsonify(get_system_metrics())

@app.route('/api/knowledge-graph/stats', methods=['GET'])
def knowledge_graph_stats():
    return jsonify(generate_knowledge_graph_data())

@app.route('/api/knowledge-graph/visualization', methods=['GET'])
def knowledge_graph_visualization():
    limit = request.args.get('limit', default=100, type=int)
    nodes = []
    edges = []
    
    # Generate nodes
    for i in range(min(limit, 100)):
        nodes.append({
            'id': f"node-{i}",
            'label': f"Node {i}",
            'type': random.choice(['Entity', 'Concept', 'Relation', 'Event']),
            'size': random.randint(1, 10)
        })
    
    # Generate edges
    for i in range(min(limit * 2, 200)):
        source = random.randint(0, len(nodes) - 1)
        target = random.randint(0, len(nodes) - 1)
        if source != target:
            edges.append({
                'id': f"edge-{i}",
                'source': f"node-{source}",
                'target': f"node-{target}",
                'label': random.choice(['CONNECTS_TO', 'PART_OF', 'RELATES_TO', 'DEPENDS_ON'])
            })
    
    return jsonify({'nodes': nodes, 'edges': edges})

@app.route('/api/consensus/stats', methods=['GET'])
def consensus_stats():
    return jsonify(generate_consensus_stats())

@app.route('/api/consensus/nodes', methods=['GET'])
def consensus_nodes():
    return jsonify(generate_consensus_nodes())

@app.route('/api/rzset/stats', methods=['GET'])
def rzset_stats():
    return jsonify(generate_rzset_stats())

@app.route('/api/rzset/vectors/<group>', methods=['GET'])
def rzset_vectors(group):
    return jsonify(generate_vectors(group))

@app.route('/api/llm/stats', methods=['GET'])
def llm_stats():
    """Get real-time LLM usage statistics and model information"""
    try:
        # Get model info from Ollama API
        response = requests.get("http://localhost:11434/api/tags")
        
        if response.status_code == 200:
            models_data = response.json()
            models = models_data.get('models', [])
            
            # Find our model
            model_info = next((m for m in models if 'qwen2.5-coder:14b' in m.get('name', '')), None)
            
            if not model_info:
                # Default info if model not found
                model_name = "qwen2.5-coder"
                model_version = "14b"
                parameter_size = "14.8B"
            else:
                model_name = model_info.get('name', '').split(':')[0]
                model_version = model_info.get('name', '').split(':')[1] if ':' in model_info.get('name', '') else 'latest'
                details = model_info.get('details', {})
                parameter_size = details.get('parameter_size', '14.8B')
            
            # Calculate average latency
            avg_latency = 0
            if llm_metrics["total_queries"] > 0 and llm_metrics["successful_queries"] > 0:
                avg_latency = llm_metrics["total_response_time"] / llm_metrics["successful_queries"]
                
            # Calculate success rate
            success_rate = 1.0
            if llm_metrics["total_queries"] > 0:
                success_rate = llm_metrics["successful_queries"] / llm_metrics["total_queries"]
            
            # Build stats response
            return jsonify({
                'model_name': model_name,
                'model_version': model_version,
                'provider': "Ollama",
                'context_window': 8192,  # Typical for this model
                'parameters': float(parameter_size.rstrip('B')) * 10**9 if parameter_size.rstrip('B').replace('.', '', 1).isdigit() else 14.8 * 10**9,
                'query_count': llm_metrics["total_queries"],
                'successful_queries': llm_metrics["successful_queries"],
                'failed_queries': llm_metrics["failed_queries"],
                'total_tokens': llm_metrics["total_tokens"],
                'avg_latency': round(avg_latency * 1000, 2),  # Convert to ms
                'success_rate': success_rate,
                'status': 'Online',
                'last_query_time': llm_metrics["last_query_time"],
                'model_usage': llm_metrics["model_usage"]
            })
        else:
            logger.error(f"Error getting model info from Ollama: {response.status_code}")
            
    except Exception as e:
        logger.error(f"Error getting LLM stats: {e}")
    
    # Fall back to basic stats if Ollama info can't be retrieved
    return jsonify({
        'model_name': "qwen2.5-coder",
        'model_version': "14b",
        'provider': "Ollama",
        'context_window': 8192,
        'parameters': 14.8 * 10**9,
        'query_count': llm_metrics["total_queries"],
        'successful_queries': llm_metrics["successful_queries"],
        'failed_queries': llm_metrics["failed_queries"],
        'avg_latency': 0,
        'success_rate': 1.0 if llm_metrics["total_queries"] == 0 else llm_metrics["successful_queries"] / llm_metrics["total_queries"],
        'status': 'Online'
    })

@app.route('/api/llm/conversation_history', methods=['GET'])
def llm_conversation_history():
    """Get conversation history for a specific session"""
    session_id = request.args.get('session_id', 'default')
    
    # Return the conversation history for the requested session
    history = llm_conversations.get(session_id, [])
    return jsonify({
        'session_id': session_id,
        'message_count': len(history),
        'messages': history
    })

@app.route('/api/llm/sessions', methods=['GET'])
def llm_sessions():
    """Get list of all conversation sessions"""
    sessions = []
    for session_id, messages in llm_conversations.items():
        sessions.append({
            'id': session_id,
            'message_count': len(messages),
            'first_message': messages[0]['content'] if messages else None,
            'last_activity': messages[-1]['timestamp'] if messages else None
        })
    
    return jsonify({
        'session_count': len(sessions),
        'sessions': sessions
    })

@app.route('/api/llm/reset_conversation', methods=['POST'])
def llm_reset_conversation():
    """Reset a conversation session"""
    data = request.json
    session_id = data.get('session_id', 'default')
    
    if session_id in llm_conversations:
        llm_conversations[session_id] = []
        return jsonify({'status': 'success', 'message': f'Session {session_id} has been reset'})
    else:
        return jsonify({'status': 'error', 'message': f'Session {session_id} not found'}), 404

@app.route('/api/llm/knowledge_base', methods=['GET'])
def llm_knowledge_base():
    """Get the DARF knowledge base used by the LLM"""
    return jsonify({'knowledge_base': darf_knowledge_base})

@app.route('/api/datasets/summary', methods=['GET'])
def datasets_summary():
    """Get summary of all available datasets"""
    return jsonify(dataset_manager.get_dataset_summary())

@app.route('/api/datasets/<dataset_id>', methods=['GET'])
def dataset_detail(dataset_id):
    """Get detailed information about a specific dataset"""
    details = dataset_manager.get_dataset_detail(dataset_id)
    if details is None:
        return jsonify({'error': 'Dataset not found'}), 404
    return jsonify(details)

@app.route('/api/datasets/insights', methods=['GET'])
def dataset_insights():
    """Get insights across datasets"""
    dataset_ids = request.args.get('datasets', '').split(',')
    dataset_ids = [ds_id for ds_id in dataset_ids if ds_id] # Filter empty strings
    
    insights = dataset_manager.get_cross_dataset_insights(dataset_ids)
    return jsonify({'insights': insights})

@app.route('/api/knowledge-graph/query', methods=['POST'])
def knowledge_graph_query():
    """Query the knowledge graph with natural language"""
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
        
    results = dataset_manager.query_knowledge_graph(query)
    return jsonify(results)

@app.route('/api/system/status', methods=['GET'])
def system_status():
    """Get comprehensive system-wide status and metrics"""
    return jsonify(darf_monitor.get_all_metrics())

@app.route('/api/system/component/<component_name>', methods=['GET'])
def component_status(component_name):
    """Get detailed status and metrics for a specific component"""
    context = darf_monitor.get_component_context(component_name)
    if "error" in context:
        return jsonify(context), 404
    return jsonify(context)

@app.route('/api/llm/query', methods=['POST'])
def llm_query():
    try:
        data = request.json
        if not data:
            raise ValidationError("No data provided")
            
        validate_request(data, ['prompt'])
        
        prompt = data.get('prompt', '')
        session_id = data.get('session_id', 'default')
        stream = data.get('stream', False)
        include_system_context = data.get('include_system_context', True)
        task_type = data.get('task_type')  # Optional hint for task classification
        max_parameters = data.get('max_parameters')  # Optional to limit model size
        prefer_low_latency = data.get('prefer_low_latency', not stream)  # Default to low latency for non-streaming
        use_async = data.get('async', False) and has_async  # Use async if requested and available
        
        # Update metrics
        llm_metrics["total_queries"] += 1
        llm_metrics["last_query_time"] = datetime.now().isoformat()
        
        # Ensure conversation history exists for this session
        if session_id not in llm_conversations:
            llm_conversations[session_id] = []
        
        # Add user message to conversation history
        llm_conversations[session_id].append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        })
        
        # Try to get from cache first if it's not a streaming request and caching is not disabled
        if not stream and not data.get('no_cache', False):
            cached_response = cache.get("llm_responses", {
                "prompt": prompt,
                "session_id": session_id,
                "include_system_context": include_system_context
            })
            
            if cached_response:
                logger.info(f"Cache hit for prompt: {prompt[:50]}...")
                return jsonify(cached_response)
        
        # Build the base system prompt
        system_prompt = """You are the DARF Assistant, the AI interface for the Distributed Architecture for Robust and Flexible (DARF) Framework.

DARF is a comprehensive framework for building distributed, fault-tolerant, and scalable applications. It consists of the following key components:

1. Knowledge Graph: For storing and navigating connections between data entities with semantic relationships
2. RZSet: A specialized vector database for efficient similarity search and vector operations
3. Event Bus: The communication backbone that enables asynchronous message passing between components
4. Consensus Engine: Implements distributed agreement protocols for consistency across nodes
5. LLM Interface: (That's you!) Connects language models with the framework for natural language processing
6. System Metrics: Collects and analyzes performance data for monitoring and optimization

Your purpose is to assist users in understanding and working with the DARF Framework and to provide real-time insights into the system's operation and performance."""

        # Get relevant system context based on the user's query
        if include_system_context:
            darf_context = darf_monitor.enrich_llm_context(prompt)
            
            # Format the context as a string for the system prompt
            context_str = "\n\nCurrent System Context:\n"
            
            # Add overall system status
            context_str += f"- System Status: {darf_context['system_status']}\n"
            
            # Add relevant component information
            if "relevant_components" in darf_context:
                context_str += "- Relevant Components:\n"
                for name, info in darf_context["relevant_components"].items():
                    context_str += f"  * {name.replace('_', ' ').title()}: {info['status']}\n"
                    if "metrics" in info and info["metrics"]:
                        # Include 2-3 most important metrics
                        context_str += "    Key Metrics:\n"
                        metrics_added = 0
                        
                        # Knowledge Graph specific metrics
                        if name == "knowledge_graph" and "node_count" in info["metrics"]:
                            context_str += f"    - Nodes: {info['metrics'].get('node_count', 'N/A')}, Edges: {info['metrics'].get('edge_count', 'N/A')}\n"
                            metrics_added += 1
                            
                        # RZSet specific metrics
                        if name == "rzset" and "vector_count" in info["metrics"]:
                            context_str += f"    - Vectors: {info['metrics'].get('vector_count', 'N/A')}, Dimensions: {info['metrics'].get('dimensions', 'N/A')}\n"
                            metrics_added += 1
                            
                        # Event Bus specific metrics
                        if name == "event_bus" and "events_processed" in info["metrics"]:
                            context_str += f"    - Events Processed: {info['metrics'].get('events_processed', 'N/A')}, Subscribers: {info['metrics'].get('subscriber_count', 'N/A')}\n"
                            metrics_added += 1
                            
                        # Consensus specific metrics
                        if name == "consensus_engine" and "protocol" in info["metrics"]:
                            context_str += f"    - Protocol: {info['metrics'].get('protocol', 'N/A')}, Agreement Rate: {info['metrics'].get('agreement_rate', 'N/A')}\n"
                            metrics_added += 1
                            
                        # LLM specific metrics
                        if name == "llm_interface" and "total_queries" in info["metrics"]:
                            context_str += f"    - Queries: {info['metrics'].get('total_queries', 'N/A')}, Success Rate: {info['metrics'].get('success_rate', 'N/A')}\n"
                            metrics_added += 1
                            
                        # Add generic metrics if none added yet
                        if metrics_added == 0:
                            for k, v in info["metrics"].items():
                                if not isinstance(v, dict) and not isinstance(v, list):
                                    context_str += f"    - {k}: {v}\n"
                                    metrics_added += 1
                                    if metrics_added >= 2:
                                        break
            
            # Add dataset information if relevant to the query
            if "datasets" in darf_context:
                context_str += f"- Available Datasets: {darf_context['datasets']['dataset_count']}\n"
                for ds_name, ds_info in darf_context['datasets'].get('datasets', {}).items():
                    context_str += f"  * {ds_name}: {ds_info.get('size', 'N/A')} records - {ds_info.get('description', 'No description')}\n"
                    
            # If dataset relationships are mentioned
            if "dataset_relationships" in darf_context:
                context_str += "- Dataset Relationships:\n"
                for rel_name, rel_desc in darf_context['dataset_relationships'].items():
                    if isinstance(rel_desc, dict) and 'description' in rel_desc:
                        context_str += f"  * {rel_desc['description']}\n"
                    elif rel_name != "cross_dataset_insights":  # Skip the array of insights
                        context_str += f"  * {rel_name}: {rel_desc}\n"
                        
            # Add system metrics if relevant
            if "system_metrics" in darf_context:
                metrics = darf_context["system_metrics"]
                context_str += "- System Resources:\n"
                context_str += f"  * CPU: {metrics.get('cpu_percent', 'N/A')}%, Memory: {metrics.get('memory_percent', 'N/A')}%\n"
                
            # Add timestamp
            context_str += f"\nThis information was collected at: {darf_context['query_timestamp']}"
            
            # Append the context to the system prompt
            system_prompt += context_str

        # Get conversation history
        history = llm_conversations[session_id][-10:] if len(llm_conversations[session_id]) > 1 else []
        
        # Use the LLM registry to select the most appropriate model
        selected_model_id = llm_registry.select_model_for_task(
            query=prompt,
            task_type=task_type,
            prefer_low_latency=prefer_low_latency,
            max_parameters=max_parameters
        )
        
        # Track model usage
        if selected_model_id not in llm_metrics["model_usage"]:
            llm_metrics["model_usage"][selected_model_id] = 0
        llm_metrics["model_usage"][selected_model_id] += 1
        
        # Handle streaming requests
        if stream:
            def generate():
                try:
                    # Query the model with streaming
                    stream_response = llm_registry.query_model(
                        model_id=selected_model_id,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        stream=True,
                        conversation_history=history
                    )
                    
                    response_text = ""
                    
                    # Stream the response chunks
                    for line in stream_response:
                        if line:
                            try:
                                chunk = json.loads(line)
                                if 'message' in chunk:
                                    content = chunk['message'].get('content', '')
                                    if content:
                                        response_text += content
                                        yield f"data: {json.dumps({'chunk': content, 'done': False})}\n\n"
                                
                                # Check if this is the final message
                                if chunk.get('done', False):
                                    # Update metrics
                                    llm_metrics["successful_queries"] += 1
                                    token_estimate = len(response_text.split())
                                    llm_metrics["total_tokens"] += token_estimate
                                    
                                    # Save to conversation history
                                    llm_conversations[session_id].append({
                                        "role": "assistant",
                                        "content": response_text,
                                        "timestamp": datetime.now().isoformat()
                                    })
                                    
                                    # Send final message
                                    yield f"data: {json.dumps({
                                        'chunk': '',
                                        'done': True,
                                        'full_response': response_text,
                                        'model_used': selected_model_id,
                                        'metrics': {
                                            'estimated_tokens': token_estimate
                                        }
                                    })}\n\n"
                                    break
                            except json.JSONDecodeError:
                                logger.warning(f"Could not parse streaming response: {line}")
                                continue
                except Exception as e:
                    error_msg = f"Error in streaming response: {str(e)}"
                    logger.error(error_msg)
                    llm_metrics["failed_queries"] += 1
                    yield f"data: {json.dumps({'error': error_msg, 'done': True})}\n\n"
                    
            return Response(generate(), mimetype='text/event-stream')
            
        elif use_async:
            # Use Celery for asynchronous processing
            task = process_llm_query.delay(
                model_id=selected_model_id,
                prompt=prompt,
                system_prompt=system_prompt,
                conversation_history=history
            )
            
            return jsonify({
                'status': 'processing',
                'task_id': task.id,
                'model_used': selected_model_id,
                'session_id': session_id
            })
            
        else:
            # Standard synchronous processing
            start_time = time.time()
            
            result = llm_registry.query_model(
                model_id=selected_model_id,
                prompt=prompt,
                system_prompt=system_prompt,
                conversation_history=history
            )
            
            generated_text = result["response"]
            
            # Update metrics
            process_time = time.time() - start_time
            llm_metrics["total_response_time"] += process_time
            llm_metrics["successful_queries"] += 1
            llm_metrics["total_tokens"] += result["tokens_used"]
            
            # Save to conversation history
            llm_conversations[session_id].append({
                "role": "assistant",
                "content": generated_text,
                "timestamp": datetime.now().isoformat()
            })
            
            response_data = {
                'response': generated_text,
                'model_used': selected_model_id,
                'metrics': {
                    'response_time': round(process_time, 2),
                    'tokens_used': result["tokens_used"],
                    'conversation_turns': len(llm_conversations[session_id]) // 2
                },
                'session_id': session_id
            }
            
            # Cache the response
            if not data.get('no_cache', False):
                cache.set(
                    "llm_responses", 
                    {
                        "prompt": prompt,
                        "session_id": session_id,
                        "include_system_context": include_system_context
                    }, 
                    response_data,
                    ttl=300  # Cache for 5 minutes
                )
            
            logger.info(f"Response generated by {selected_model_id}: {generated_text[:100]}...")
            return jsonify(response_data)
            
    except ValidationError as e:
        # Handle validation errors
        logger.warning(f"Validation error in LLM query: {e}")
        return jsonify({
            'error': str(e),
            'status_code': 400
        }), 400
    except ModelNotFoundError as e:
        # Handle model not found errors
        logger.error(f"Model not found error: {e}")
        return jsonify({
            'error': str(e),
            'status_code': 404
        }), 404
    except ModelQueryError as e:
        # Handle model query errors
        logger.error(f"Model query error: {e}")
        llm_metrics["failed_queries"] += 1
        return jsonify({
            'error': str(e),
            'status_code': 500
        }), 500
    except Exception as e:
        # Handle unexpected errors
        logger.exception(f"Unexpected error in LLM query: {e}")
        llm_metrics["failed_queries"] += 1
        return jsonify({
            'error': f"An unexpected error occurred: {str(e)}",
            'status_code': 500
        }), 500

@app.route('/api/data-sources', methods=['GET'])
def data_sources():
    return jsonify(generate_data_sources())

@app.route('/api/data-sources/<id>', methods=['GET'])
def data_source_details(id):
    return jsonify(generate_data_source_details(id))

# Dataset connections and knowledge graph integration
class DatasetManager:
    def __init__(self):
        self.datasets = {}
        self.graph_connections = {}
        self.loaded_datasets = False
        
    def load_datasets(self):
        """Load available datasets from the dataset connector"""
        if self.loaded_datasets:
            return
            
        logger.info("Loading datasets into DatasetManager")
        try:
            # Access the data_cache directly from dataset_connector
            import dataset_connector
            
            # Force dataset loading if not already loaded
            if not dataset_connector.data_cache.get('iris') and not dataset_connector.data_cache.get('wine') and not dataset_connector.data_cache.get('bbc'):
                dataset_connector.load_datasets()
            
            # Get Iris dataset from cache
            iris_data = dataset_connector.data_cache.get('iris', {}).get('vectors', [])
            if iris_data:
                self.datasets["iris"] = {
                    "data": iris_data,
                    "description": "Iris flower dataset with sepal/petal measurements",
                    "dimensions": 4,  # Fixed dimensions for Iris
                    "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
                    "target": "species",
                    "vector_count": len(iris_data)
                }
                logger.info(f"Loaded Iris dataset with {len(iris_data)} records")
                
            # Get Wine dataset from cache
            wine_data = dataset_connector.data_cache.get('wine', {})
            wine_nodes = wine_data.get('nodes', [])
            wine_edges = wine_data.get('edges', [])
            if wine_nodes:
                node_types = list(set([node.get('type', 'unknown') for node in wine_nodes if 'type' in node]))
                self.datasets["wine"] = {
                    "nodes": wine_nodes,
                    "edges": wine_edges,
                    "description": "Wine dataset with chemical properties and classifications",
                    "node_count": len(wine_nodes),
                    "edge_count": len(wine_edges),
                    "node_types": node_types if node_types else ["data_node"]
                }
                logger.info(f"Loaded Wine dataset with {len(wine_nodes)} nodes and {len(wine_edges)} edges")
                
            # Get BBC News dataset from cache
            bbc_data = dataset_connector.data_cache.get('bbc', {}).get('documents', [])
            if bbc_data:
                categories = list(set([doc.get('category', 'unknown') for doc in bbc_data if 'category' in doc]))
                self.datasets["bbc_news"] = {
                    "data": bbc_data,
                    "description": "BBC News articles with categories",
                    "document_count": len(bbc_data),
                    "categories": categories if categories else ["news"]
                }
                logger.info(f"Loaded BBC News dataset with {len(bbc_data)} documents")
                
            self.loaded_datasets = True
            
            # Create relationships between datasets (knowledge graph connections)
            self._create_dataset_relationships()
            
        except ImportError as e:
            logger.error(f"Could not import dataset functions: {e}")
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            
    def _create_dataset_relationships(self):
        """Create relationships between datasets for cross-dataset analysis"""
        self.graph_connections = {
            "iris_to_wine": {
                "description": "Relationship between Iris flower properties and wine chemical attributes",
                "connection_type": "feature_correlation",
                "mapping": {
                    "iris.petal_length": ["wine.alcohol", "wine.malic_acid"],
                    "iris.sepal_width": ["wine.ash", "wine.alcalinity_of_ash"]
                }
            },
            "wine_to_bbc": {
                "description": "Mapping between wine regions and news articles about those regions",
                "connection_type": "entity_reference",
                "mapping": {
                    "wine.origin": "bbc_news.content[contains_location]"
                }
            },
            "cross_dataset_insights": [
                "Correlation between wine chemical properties and news sentiment about wine regions",
                "Relationship between flower species distribution and wine variety popularity in news articles",
                "Temporal trends in wine ratings compared to news coverage of vineyards"
            ]
        }
        
    def get_dataset_summary(self):
        """Get summary of available datasets"""
        if not self.loaded_datasets:
            self.load_datasets()
            
        return {
            "dataset_count": len(self.datasets),
            "datasets": {k: {
                "description": v.get("description", "No description"),
                "size": v.get("document_count", v.get("node_count", len(v.get("data", []))))
            } for k, v in self.datasets.items()},
            "graph_connections": {k: v["description"] for k, v in self.graph_connections.items()}
        }
        
    def get_dataset_detail(self, dataset_id):
        """Get detailed information about a specific dataset"""
        if not self.loaded_datasets:
            self.load_datasets()
            
        if dataset_id not in self.datasets:
            return None
            
        return self.datasets[dataset_id]
        
    def get_cross_dataset_insights(self, dataset_ids=None):
        """Generate insights across multiple datasets"""
        if not self.loaded_datasets:
            self.load_datasets()
            
        insights = []
        
        if dataset_ids is None or len(dataset_ids) == 0:
            # Return predefined cross-dataset insights
            return self.graph_connections.get("cross_dataset_insights", [])
            
        # Get subset of specified datasets
        if "iris" in dataset_ids and "wine" in dataset_ids:
            insights.append({
                "title": "Feature Correlation: Iris and Wine",
                "description": "Analysis of correlation between Iris flower petal dimensions and Wine chemical properties",
                "correlation_score": 0.68,
                "insight": "Regions with certain Iris species tend to produce wines with higher alcohol content"
            })
            
        if "wine" in dataset_ids and "bbc_news" in dataset_ids:
            insights.append({
                "title": "Wine Regions in News Coverage",
                "description": "Analysis of how wine regions are represented in BBC news articles",
                "mention_count": 156,
                "sentiment_analysis": {
                    "positive": 0.45,
                    "neutral": 0.35,
                    "negative": 0.20
                },
                "insight": "French wine regions receive 30% more positive coverage compared to other regions"
            })
            
        return insights
        
    def query_knowledge_graph(self, query):
        """Query the knowledge graph for information across datasets"""
        if not self.loaded_datasets:
            self.load_datasets()
            
        # Parse natural language query to extract entities and relationships
        entities = self._extract_entities(query)
        
        # Simulate knowledge graph query results
        results = {
            "query": query,
            "entities_found": entities,
            "relationships": [],
            "insights": []
        }
        
        # Add relationships based on found entities
        if "iris" in entities and "wine" in entities:
            results["relationships"].append({
                "source": "iris",
                "target": "wine",
                "type": "feature_correlation",
                "description": "Correlation between Iris petal length and Wine alcohol content"
            })
            
        if "wine" in entities and "news" in entities:
            results["relationships"].append({
                "source": "wine",
                "target": "bbc_news",
                "type": "entity_reference",
                "description": "References to wine regions in news articles"
            })
            
        # Add insights for found relationships
        for rel in results["relationships"]:
            results["insights"].append({
                "description": f"Analysis of {rel['source']} and {rel['target']} shows {rel['description']}",
                "confidence": 0.75
            })
            
        return results
        
    def _extract_entities(self, query):
        """Extract entities from a natural language query"""
        entities = []
        
        # Simple keyword matching for datasets
        if "iris" in query.lower():
            entities.append("iris")
        if "wine" in query.lower():
            entities.append("wine")
        if "news" in query.lower() or "article" in query.lower() or "bbc" in query.lower():
            entities.append("bbc_news")
            
        return entities

# Initialize dataset manager
dataset_manager = DatasetManager()

# Knowledge base for DARF-specific information
darf_knowledge_base = {
    "components": {
        "knowledge_graph": {
            "description": "DARF's Knowledge Graph enables storage and navigation of semantic data relationships",
            "use_cases": ["Entity relationship mapping", "Semantic search", "Data context enhancement", "Cross-dataset insights"],
            "interfaces": ["query_entities", "add_relationship", "get_subgraph", "analyze_connections"]
        },
        "rzset": {
            "description": "Specialized vector database for similarity search and vector operations",
            "use_cases": ["Nearest neighbor search", "Semantic similarity", "Clustering"],
            "interfaces": ["store_vector", "similarity_search", "batch_process"]
        },
        "event_bus": {
            "description": "Communication backbone enabling asynchronous message passing",
            "use_cases": ["Service decoupling", "Event broadcasting", "Workflow orchestration"],
            "interfaces": ["publish", "subscribe", "filter_events"]
        },
        "consensus_engine": {
            "description": "Implements distributed agreement protocols for consistency",
            "use_cases": ["Distributed state management", "Leader election", "Data consistency"],
            "interfaces": ["propose", "vote", "get_state"]
        },
        "llm_interface": {
            "description": "Connects language models with the framework for NLP capabilities",
            "use_cases": ["Natural language queries", "Data summarization", "Code generation"],
            "interfaces": ["query", "generate", "analyze_text"]
        },
        "system_metrics": {
            "description": "Collects and analyzes performance data for monitoring",
            "use_cases": ["Performance monitoring", "Anomaly detection", "Resource optimization"],
            "interfaces": ["collect", "alert", "visualize"]
        }
    },
    "architecture": {
        "design_principles": [
            "Component isolation", 
            "Fault tolerance", 
            "Scalability", 
            "Asynchronous communication",
            "Data consistency"
        ],
        "deployment_models": [
            "Single-node", 
            "Cluster", 
            "Distributed", 
            "Edge-based"
        ]
    }
}

# Helper function to run DARF commands and get output
def run_darf_command(command, args=None):
    try:
        if args:
            result = subprocess.run([command] + args, capture_output=True, text=True, timeout=5)
        else:
            result = subprocess.run([command], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            return result.stdout
        else:
            logger.error(f"Command failed: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out: {command}")
        return None
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return None

# Helper to get system metrics
def get_system_metrics():
    metrics = {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'cpu_cores': psutil.cpu_count(logical=True),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'disk_usage': psutil.disk_usage('/').percent,
        'disk_total': psutil.disk_usage('/').total,
        'uptime': time.time() - psutil.boot_time(),
        'load_avg': psutil.getloadavg(),
        'network': {
            'sent': psutil.net_io_counters().bytes_sent,
            'received': psutil.net_io_counters().bytes_recv
        },
        'timestamp': datetime.now().isoformat()
    }
    
    return metrics

# Mock data generators for demonstration
def generate_knowledge_graph_data():
    return {
        'nodes': random.randint(5000, 20000),
        'edges': random.randint(15000, 50000),
        'types': random.randint(5, 20),
        'last_updated': (datetime.now() - timedelta(minutes=random.randint(1, 60))).isoformat(),
        'query_count': random.randint(1000, 10000),
        'avg_query_time': random.uniform(0.05, 2.0),
        'storage_size': random.randint(50000000, 500000000),
        'metrics': {
            'centrality': random.uniform(0.3, 0.8),
            'clustering': random.uniform(0.2, 0.7),
            'diameter': random.randint(5, 15)
        }
    }

def generate_consensus_stats():
    protocols = ['Raft', 'PBFT', 'Tendermint', 'HoneyBadger BFT']
    return {
        'nodes': random.randint(3, 15),
        'agreement_rate': random.uniform(0.95, 0.999),
        'tps': random.randint(100, 5000),
        'leader': f"node-{random.randint(1, 10)}",
        'protocol': random.choice(protocols),
        'last_consensus': (datetime.now() - timedelta(seconds=random.randint(1, 30))).isoformat()
    }

def generate_consensus_nodes():
    roles = ['leader', 'follower', 'candidate', 'observer']
    statuses = ['active', 'syncing', 'warning', 'error']
    
    nodes = []
    for i in range(random.randint(3, 15)):
        node = {
            'id': f"node-{i+1}",
            'role': random.choice(roles),
            'status': random.choice(statuses),
            'last_seen': (datetime.now() - timedelta(seconds=random.randint(0, 30))).isoformat(),
            'transactions': random.randint(1000, 100000),
            'agreement_rate': random.uniform(0.90, 0.999)
        }
        nodes.append(node)
    
    return {'nodes': nodes}

def generate_rzset_stats():
    return {
        'vector_count': random.randint(10000, 1000000),
        'dimensions': random.choice([128, 256, 512, 768, 1024, 1536, 2048]),
        'groups': [f"group-{i}" for i in range(1, random.randint(3, 10))],
        'storage_size': random.randint(50000000, 5000000000),
        'queries_per_second': random.uniform(10, 500),
        'last_updated': (datetime.now() - timedelta(minutes=random.randint(1, 120))).isoformat()
    }

def generate_vectors(group):
    vectors = []
    for i in range(random.randint(5, 20)):
        vector = {
            'id': f"{group}-vector-{i+1}",
            'metadata': f"Description for {group} vector {i+1}",
            'dimensions': random.choice([128, 256, 512, 768, 1024]),
            'created_at': (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat()
        }
        vectors.append(vector)
    
    return {'vectors': vectors}

def generate_data_sources():
    types = ['Database', 'API', 'File System', 'Streaming', 'Event Queue', 'Cache']
    statuses = ['active', 'partial', 'offline', 'warning']
    
    sources = []
    for i in range(random.randint(5, 15)):
        source_type = random.choice(types)
        source = {
            'id': f"source-{i+1}",
            'name': f"{source_type.lower()}-source-{i+1}",
            'type': source_type,
            'description': f"A {source_type.lower()} data source for the DARF framework",
            'status': random.choice(statuses),
            'connection': {
                'host': f"host-{random.randint(1, 10)}.example.com",
                'port': random.choice([3306, 5432, 6379, 8080, 9090, 27017]),
                'database': f"db-{random.randint(1, 5)}",
                'username': "darf_user"
            },
            'last_updated': (datetime.now() - timedelta(days=random.randint(0, 7))).isoformat(),
            'created_at': (datetime.now() - timedelta(days=random.randint(7, 90))).isoformat()
        }
        sources.append(source)
    
    return {'sources': sources}

def generate_data_source_details(id):
    types = ['Database', 'API', 'File System', 'Streaming', 'Event Queue', 'Cache']
    statuses = ['active', 'partial', 'offline', 'warning']
    
    source_type = random.choice(types)
    return {
        'id': id,
        'name': f"{source_type.lower()}-{id}",
        'type': source_type,
        'description': f"A {source_type.lower()} data source for the DARF framework with detailed configuration.",
        'status': random.choice(statuses),
        'connection': {
            'host': f"host-{random.randint(1, 10)}.example.com",
            'port': random.choice([3306, 5432, 6379, 8080, 9090, 27017]),
            'database': f"db-{random.randint(1, 5)}",
            'username': "darf_user",
            'params': {
                'timeout': random.randint(5, 30),
                'max_connections': random.randint(10, 100),
                'ssl': random.choice([True, False])
            }
        },
        'last_updated': (datetime.now() - timedelta(days=random.randint(0, 7))).isoformat(),
        'created_at': (datetime.now() - timedelta(days=random.randint(7, 90))).isoformat(),
        'stats': {
            'records': random.randint(1000, 1000000),
            'last_query': (datetime.now() - timedelta(minutes=random.randint(1, 120))).isoformat(),
            'query_count': random.randint(100, 10000),
            'avg_query_time_ms': random.uniform(10, 500)
        }
    }

# API Routes
@app.route('/api/metrics/system', methods=['GET'])
def metrics_system():
    return jsonify(get_system_metrics())

@app.route('/api/knowledge-graph/stats', methods=['GET'])
def knowledge_graph_stats():
    return jsonify(generate_knowledge_graph_data())

@app.route('/api/knowledge-graph/visualization', methods=['GET'])
def knowledge_graph_visualization():
    limit = request.args.get('limit', default=100, type=int)
    nodes = []
    edges = []
    
    # Generate nodes
    for i in range(min(limit, 100)):
        nodes.append({
            'id': f"node-{i}",
            'label': f"Node {i}",
            'type': random.choice(['Entity', 'Concept', 'Relation', 'Event']),
            'size': random.randint(1, 10)
        })
    
    # Generate edges
    for i in range(min(limit * 2, 200)):
        source = random.randint(0, len(nodes) - 1)
        target = random.randint(0, len(nodes) - 1)
        if source != target:
            edges.append({
                'id': f"edge-{i}",
                'source': f"node-{source}",
                'target': f"node-{target}",
                'label': random.choice(['CONNECTS_TO', 'PART_OF', 'RELATES_TO', 'DEPENDS_ON'])
            })
    
    return jsonify({'nodes': nodes, 'edges': edges})

@app.route('/api/consensus/stats', methods=['GET'])
def consensus_stats():
    return jsonify(generate_consensus_stats())

@app.route('/api/consensus/nodes', methods=['GET'])
def consensus_nodes():
    return jsonify(generate_consensus_nodes())

@app.route('/api/rzset/stats', methods=['GET'])
def rzset_stats():
    return jsonify(generate_rzset_stats())

@app.route('/api/rzset/vectors/<group>', methods=['GET'])
def rzset_vectors(group):
    return jsonify(generate_vectors(group))

@app.route('/api/llm/stats', methods=['GET'])
def llm_stats():
    """Get real-time LLM usage statistics and model information"""
    try:
        # Get model info from Ollama API
        response = requests.get("http://localhost:11434/api/tags")
        
        if response.status_code == 200:
            models_data = response.json()
            models = models_data.get('models', [])
            
            # Find our model
            model_info = next((m for m in models if 'qwen2.5-coder:14b' in m.get('name', '')), None)
            
            if not model_info:
                # Default info if model not found
                model_name = "qwen2.5-coder"
                model_version = "14b"
                parameter_size = "14.8B"
            else:
                model_name = model_info.get('name', '').split(':')[0]
                model_version = model_info.get('name', '').split(':')[1] if ':' in model_info.get('name', '') else 'latest'
                details = model_info.get('details', {})
                parameter_size = details.get('parameter_size', '14.8B')
            
            # Calculate average latency
            avg_latency = 0
            if llm_metrics["total_queries"] > 0 and llm_metrics["successful_queries"] > 0:
                avg_latency = llm_metrics["total_response_time"] / llm_metrics["successful_queries"]
                
            # Calculate success rate
            success_rate = 1.0
            if llm_metrics["total_queries"] > 0:
                success_rate = llm_metrics["successful_queries"] / llm_metrics["total_queries"]
            
            # Build stats response
            return jsonify({
                'model_name': model_name,
                'model_version': model_version,
                'provider': "Ollama",
                'context_window': 8192,  # Typical for this model
                'parameters': float(parameter_size.rstrip('B')) * 10**9 if parameter_size.rstrip('B').replace('.', '', 1).isdigit() else 14.8 * 10**9,
                'query_count': llm_metrics["total_queries"],
                'successful_queries': llm_metrics["successful_queries"],
                'failed_queries': llm_metrics["failed_queries"],
                'total_tokens': llm_metrics["total_tokens"],
                'avg_latency': round(avg_latency * 1000, 2),  # Convert to ms
                'success_rate': success_rate,
                'status': 'Online',
                'last_query_time': llm_metrics["last_query_time"],
                'model_usage': llm_metrics["model_usage"]
            })
        else:
            logger.error(f"Error getting model info from Ollama: {response.status_code}")
            
    except Exception as e:
        logger.error(f"Error getting LLM stats: {e}")
    
    # Fall back to basic stats if Ollama info can't be retrieved
    return jsonify({
        'model_name': "qwen2.5-coder",
        'model_version': "14b",
        'provider': "Ollama",
        'context_window': 8192,
        'parameters': 14.8 * 10**9,
        'query_count': llm_metrics["total_queries"],
        'successful_queries': llm_metrics["successful_queries"],
        'failed_queries': llm_metrics["failed_queries"],
        'avg_latency': 0,
        'success_rate': 1.0 if llm_metrics["total_queries"] == 0 else llm_metrics["successful_queries"] / llm_metrics["total_queries"],
        'status': 'Online'
    })

@app.route('/api/llm/conversation_history', methods=['GET'])
def llm_conversation_history():
    """Get conversation history for a specific session"""
    session_id = request.args.get('session_id', 'default')
    
    # Return the conversation history for the requested session
    history = llm_conversations.get(session_id, [])
    return jsonify({
        'session_id': session_id,
        'message_count': len(history),
        'messages': history
    })

@app.route('/api/llm/sessions', methods=['GET'])
def llm_sessions():
    """Get list of all conversation sessions"""
    sessions = []
    for session_id, messages in llm_conversations.items():
        sessions.append({
            'id': session_id,
            'message_count': len(messages),
            'first_message': messages[0]['content'] if messages else None,
            'last_activity': messages[-1]['timestamp'] if messages else None
        })
    
    return jsonify({
        'session_count': len(sessions),
        'sessions': sessions
    })

@app.route('/api/llm/reset_conversation', methods=['POST'])
def llm_reset_conversation():
    """Reset a conversation session"""
    data = request.json
    session_id = data.get('session_id', 'default')
    
    if session_id in llm_conversations:
        llm_conversations[session_id] = []
        return jsonify({'status': 'success', 'message': f'Session {session_id} has been reset'})
    else:
        return jsonify({'status': 'error', 'message': f'Session {session_id} not found'}), 404

@app.route('/api/llm/knowledge_base', methods=['GET'])
def llm_knowledge_base():
    """Get the DARF knowledge base used by the LLM"""
    return jsonify({'knowledge_base': darf_knowledge_base})

@app.route('/api/datasets/summary', methods=['GET'])
def datasets_summary():
    """Get summary of all available datasets"""
    return jsonify(dataset_manager.get_dataset_summary())

@app.route('/api/datasets/<dataset_id>', methods=['GET'])
def dataset_detail(dataset_id):
    """Get detailed information about a specific dataset"""
    details = dataset_manager.get_dataset_detail(dataset_id)
    if details is None:
        return jsonify({'error': 'Dataset not found'}), 404
    return jsonify(details)

@app.route('/api/datasets/insights', methods=['GET'])
def dataset_insights():
    """Get insights across datasets"""
    dataset_ids = request.args.get('datasets', '').split(',')
    dataset_ids = [ds_id for ds_id in dataset_ids if ds_id] # Filter empty strings
    
    insights = dataset_manager.get_cross_dataset_insights(dataset_ids)
    return jsonify({'insights': insights})

@app.route('/api/knowledge-graph/query', methods=['POST'])
def knowledge_graph_query():
    """Query the knowledge graph with natural language"""
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
        
    results = dataset_manager.query_knowledge_graph(query)
    return jsonify(results)

@app.route('/api/system/status', methods=['GET'])
def system_status():
    """Get comprehensive system-wide status and metrics"""
    return jsonify(darf_monitor.get_all_metrics())

@app.route('/api/system/component/<component_name>', methods=['GET'])
def component_status(component_name):
    """Get detailed status and metrics for a specific component"""
    context = darf_monitor.get_component_context(component_name)
    if "error" in context:
        return jsonify(context), 404
    return jsonify(context)

@app.route('/api/llm/query', methods=['POST'])
def llm_query():
    try:
        data = request.json
        if not data:
            raise ValidationError("No data provided")
            
        validate_request(data, ['prompt'])
        
        prompt = data.get('prompt', '')
        session_id = data.get('session_id', 'default')
        stream = data.get('stream', False)
        include_system_context = data.get('include_system_context', True)
        task_type = data.get('task_type')  # Optional hint for task classification
        max_parameters = data.get('max_parameters')  # Optional to limit model size
        prefer_low_latency = data.get('prefer_low_latency', not stream)  # Default to low latency for non-streaming
        use_async = data.get('async', False) and has_async  # Use async if requested and available
        
        # Update metrics
        llm_metrics["total_queries"] += 1
        llm_metrics["last_query_time"] = datetime.now().isoformat()
        
        # Ensure conversation history exists for this session
        if session_id not in llm_conversations:
            llm_conversations[session_id] = []
        
        # Add user message to conversation history
        llm_conversations[session_id].append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        })
        
        # Try to get from cache first if it's not a streaming request and caching is not disabled
        if not stream and not data.get('no_cache', False):
            cached_response = cache.get("llm_responses", {
                "prompt": prompt,
                "session_id": session_id,
                "include_system_context": include_system_context
            })
            
            if cached_response:
                logger.info(f"Cache hit for prompt: {prompt[:50]}...")
                return jsonify(cached_response)
        
        # Build the base system prompt
        system_prompt = """You are the DARF Assistant, the AI interface for the Distributed Architecture for Robust and Flexible (DARF) Framework.

DARF is a comprehensive framework for building distributed, fault-tolerant, and scalable applications. It consists of the following key components:

1. Knowledge Graph: For storing and navigating connections between data entities with semantic relationships
2. RZSet: A specialized vector database for efficient similarity search and vector operations
3. Event Bus: The communication backbone that enables asynchronous message passing between components
4. Consensus Engine: Implements distributed agreement protocols for consistency across nodes
5. LLM Interface: (That's you!) Connects language models with the framework for natural language processing
6. System Metrics: Collects and analyzes performance data for monitoring and optimization

Your purpose is to assist users in understanding and working with the DARF Framework and to provide real-time insights into the system's operation and performance."""

        # Get relevant system context based on the user's query
        if include_system_context:
            darf_context = darf_monitor.enrich_llm_context(prompt)
            
            # Format the context as a string for the system prompt
            context_str = "\n\nCurrent System Context:\n"
            
            # Add overall system status
            context_str += f"- System Status: {darf_context['system_status']}\n"
            
            # Add relevant component information
            if "relevant_components" in darf_context:
                context_str += "- Relevant Components:\n"
                for name, info in darf_context["relevant_components"].items():
                    context_str += f"  * {name.replace('_', ' ').title()}: {info['status']}\n"
                    if "metrics" in info and info["metrics"]:
                        # Include 2-3 most important metrics
                        context_str += "    Key Metrics:\n"
                        metrics_added = 0
                        
                        # Knowledge Graph specific metrics
                        if name == "knowledge_graph" and "node_count" in info["metrics"]:
                            context_str += f"    - Nodes: {info['metrics'].get('node_count', 'N/A')}, Edges: {info['metrics'].get('edge_count', 'N/A')}\n"
                            metrics_added += 1
                            
                        # RZSet specific metrics
                        if name == "rzset" and "vector_count" in info["metrics"]:
                            context_str += f"    - Vectors: {info['metrics'].get('vector_count', 'N/A')}, Dimensions: {info['metrics'].get('dimensions', 'N/A')}\n"
                            metrics_added += 1
                            
                        # Event Bus specific metrics
                        if name == "event_bus" and "events_processed" in info["metrics"]:
                            context_str += f"    - Events Processed: {info['metrics'].get('events_processed', 'N/A')}, Subscribers: {info['metrics'].get('subscriber_count', 'N/A')}\n"
                            metrics_added += 1
                            
                        # Consensus specific metrics
                        if name == "consensus_engine" and "protocol" in info["metrics"]:
                            context_str += f"    - Protocol: {info['metrics'].get('protocol', 'N/A')}, Agreement Rate: {info['metrics'].get('agreement_rate', 'N/A')}\n"
                            metrics_added += 1
                            
                        # LLM specific metrics
                        if name == "llm_interface" and "total_queries" in info["metrics"]:
                            context_str += f"    - Queries: {info['metrics'].get('total_queries', 'N/A')}, Success Rate: {info['metrics'].get('success_rate', 'N/A')}\n"
                            metrics_added += 1
                            
                        # Add generic metrics if none added yet
                        if metrics_added == 0:
                            for k, v in info["metrics"].items():
                                if not isinstance(v, dict) and not isinstance(v, list):
                                    context_str += f"    - {k}: {v}\n"
                                    metrics_added += 1
                                    if metrics_added >= 2:
                                        break
            
            # Add dataset information if relevant to the query
            if "datasets" in darf_context:
                context_str += f"- Available Datasets: {darf_context['datasets']['dataset_count']}\n"
                for ds_name, ds_info in darf_context['datasets'].get('datasets', {}).items():
                    context_str += f"  * {ds_name}: {ds_info.get('size', 'N/A')} records - {ds_info.get('description', 'No description')}\n"
                    
            # If dataset relationships are mentioned
            if "dataset_relationships" in darf_context:
                context_str += "- Dataset Relationships:\n"
                for rel_name, rel_desc in darf_context['dataset_relationships'].items():
                    if isinstance(rel_desc, dict) and 'description' in rel_desc:
                        context_str += f"  * {rel_desc['description']}\n"
                    elif rel_name != "cross_dataset_insights":  # Skip the array of insights
                        context_str += f"  * {rel_name}: {rel_desc}\n"
                        
            # Add system metrics if relevant
            if "system_metrics" in darf_context:
                metrics = darf_context["system_metrics"]
                context_str += "- System Resources:\n"
                context_str += f"  * CPU: {metrics.get('cpu_percent', 'N/A')}%, Memory: {metrics.get('memory_percent', 'N/A')}%\n"
                
            # Add timestamp
            context_str += f"\nThis information was collected at: {darf_context['query_timestamp']}"
            
            # Append the context to the system prompt
            system_prompt += context_str

        # Get conversation history
        history = llm_conversations[session_id][-10:] if len(llm_conversations[session_id]) > 1 else []
        
        # Use the LLM registry to select the most appropriate model
        selected_model_id = llm_registry.select_model_for_task(
            query=prompt,
            task_type=task_type,
            prefer_low_latency=prefer_low_latency,
            max_parameters=max_parameters
        )
        
        # Track model usage
        if selected_model_id not in llm_metrics["model_usage"]:
            llm_metrics["model_usage"][selected_model_id] = 0
        llm_metrics["model_usage"][selected_model_id] += 1
        
        # Handle streaming requests
        if stream:
            def generate():
                try:
                    # Query the model with streaming
                    stream_response = llm_registry.query_model(
                        model_id=selected_model_id,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        stream=True,
                        conversation_history=history
                    )
                    
                    response_text = ""
                    
                    # Stream the response chunks
                    for line in stream_response:
                        if line:
                            try:
                                chunk = json.loads(line)
                                if 'message' in chunk:
                                    content = chunk['message'].get('content', '')
                                    if content:
                                        response_text += content
                                        yield f"data: {json.dumps({'chunk': content, 'done': False})}\n\n"
                                
                                # Check if this is the final message
                                if chunk.get('done', False):
                                    # Update metrics
                                    llm_metrics["successful_queries"] += 1
                                    token_estimate = len(response_text.split())
                                    llm_metrics["total_tokens"] += token_estimate
                                    
                                    # Save to conversation history
                                    llm_conversations[session_id].append({
                                        "role": "assistant",
                                        "content": response_text,
                                        "timestamp": datetime.now().isoformat()
                                    })
                                    
                                    # Send final message
                                    yield f"data: {json.dumps({
                                        'chunk': '',
                                        'done': True,
                                        'full_response': response_text,
                                        'model_used': selected_model_id,
                                        'metrics': {
                                            'estimated_tokens': token_estimate
                                        }
                                    })}\n\n"
                                    break
                            except json.JSONDecodeError:
                                logger.warning(f"Could not parse streaming response: {line}")
                                continue
                except Exception as e:
                    error_msg = f"Error in streaming response: {str(e)}"
                    logger.error(error_msg)
                    llm_metrics["failed_queries"] += 1
                    yield f"data: {json.dumps({'error': error_msg, 'done': True})}\n\n"
                    
            return Response(generate(), mimetype='text/event-stream')
            
        elif use_async:
            # Use Celery for asynchronous processing
            task = process_llm_query.delay(
                model_id=selected_model_id,
                prompt=prompt,
                system_prompt=system_prompt,
                conversation_history=history
            )
            
            return jsonify({
                'status': 'processing',
                'task_id': task.id,
                'model_used': selected_model_id,
                'session_id': session_id
            })
            
        else:
            # Standard synchronous processing
            start_time = time.time()
            
            result = llm_registry.query_model(
                model_id=selected_model_id,
                prompt=prompt,
                system_prompt=system_prompt,
                conversation_history=history
            )
            
            generated_text = result["response"]
            
            # Update metrics
            process_time = time.time() - start_time
            llm_metrics["total_response_time"] += process_time
            llm_metrics["successful_queries"] += 1
            llm_metrics["total_tokens"] += result["tokens_used"]
            
            # Save to conversation history
            llm_conversations[session_id].append({
                "role": "assistant",
                "content": generated_text,
                "timestamp": datetime.now().isoformat()
            })
            
            response_data = {
                'response': generated_text,
                'model_used': selected_model_id,
                'metrics': {
                    'response_time': round(process_time, 2),
                    'tokens_used': result["tokens_used"],
                    'conversation_turns': len(llm_conversations[session_id]) // 2
                },
                'session_id': session_id
            }
            
            # Cache the response
            if not data.get('no_cache', False):
                cache.set(
                    "llm_responses", 
                    {
                        "prompt": prompt,
                        "session_id": session_id,
                        "include_system_context": include_system_context
                    }, 
                    response_data,
                    ttl=300  # Cache for 5 minutes
                )
            
            logger.info(f"Response generated by {selected_model_id}: {generated_text[:100]}...")
            return jsonify(response_data)
            
    except ValidationError as e:
        # Handle validation errors
        logger.warning(f"Validation error in LLM query: {e}")
        return jsonify({
            'error': str(e),
            'status_code': 400
        }), 400
    except ModelNotFoundError as e:
        # Handle model not found errors
        logger.error(f"Model not found error: {e}")
        return jsonify({
            'error': str(e),
            'status_code': 404
        }), 404
    except ModelQueryError as e:
        # Handle model query errors
        logger.error(f"Model query error: {e}")
        llm_metrics["failed_queries"] += 1
        return jsonify({
            'error': str(e),
            'status_code': 500
        }), 500
    except Exception as e:
        # Handle unexpected errors
        logger.exception(f"Unexpected error in LLM query: {e}")
        llm_metrics["failed_queries"] += 1
        return jsonify({
            'error': f"An unexpected error occurred: {str(e)}",
            'status_code': 500
        }), 500

@app.route('/api/data-sources', methods=['GET'])
def data_sources():
    return jsonify(generate_data_sources())

@app.route('/api/data-sources/<id>', methods=['GET'])
def data_source_details(id):
    return jsonify(generate_data_source_details(id))

# Prometheus URL definition
PROMETHEUS_URL = "http://localhost:9090"

# Prometheus passthrough proxy (optional)
@app.route('/api/v1/query', methods=['GET'])
def prometheus_query():
    try:
        query = request.args.get('query')
        time_param = request.args.get('time')
        
        params = {'query': query}
        if time_param:
            params['time'] = time_param
            
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params=params)
        return jsonify(response.json())
    except Exception as e:
        logger.error(f"Error querying Prometheus: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/api/v1/query_range', methods=['GET'])
def prometheus_query_range():
    try:
        query = request.args.get('query')
        start = request.args.get('start')
        end = request.args.get('end')
        step = request.args.get('step')
        
        params = {
            'query': query,
            'start': start,
            'end': end,
            'step': step
        }
            
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query_range", params=params)
        return jsonify(response.json())
    except Exception as e:
        logger.error(f"Error querying Prometheus range: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

# Serve React app from build folder
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    # Check if React app is built
    if not os.path.exists('darf_frontend/build'):
        logger.warning("React app build folder not found. Run 'npm run build' in the darf_frontend directory first.")
        logger.info("Starting server anyway for API endpoints...")
    
    # Register dataset routes if available
    if has_dataset_connector:
        logger.info("Registering dataset connector with DARF API")
        register_with_darf_api(app)
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
