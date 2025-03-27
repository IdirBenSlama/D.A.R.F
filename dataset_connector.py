import os
import json
import logging
import time
from pathlib import Path
from flask import Flask, jsonify, request, Blueprint
import pandas as pd
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path to datasets
datasets_dir = Path("datasets")
processed_dir = datasets_dir / "processed"

# Blueprint for dataset API routes
dataset_bp = Blueprint('datasets', __name__)

# Cache for loaded data
data_cache = {
    'iris': None,
    'wine': None,
    'bbc': None,
    'info': None,
    'last_loaded': None
}

def load_datasets():
    """Load all processed datasets into memory"""
    try:
        logger.info("Loading datasets into memory")
        
        # Load dataset info
        info_path = datasets_dir / "dataset_info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                data_cache['info'] = json.load(f)
        
        # Load Iris dataset
        iris_path = processed_dir / "iris_vectors.json"
        if iris_path.exists():
            with open(iris_path, 'r') as f:
                data_cache['iris'] = json.load(f)
                logger.info(f"Loaded {len(data_cache['iris'].get('vectors', []))} Iris vectors")
        
        # Load Wine dataset
        wine_path = processed_dir / "wine_knowledge_graph.json"
        if wine_path.exists():
            with open(wine_path, 'r') as f:
                data_cache['wine'] = json.load(f)
                logger.info(f"Loaded {len(data_cache['wine'].get('nodes', []))} Wine nodes and "
                           f"{len(data_cache['wine'].get('edges', []))} edges")
        
        # Load BBC News dataset
        bbc_path = processed_dir / "bbc_news_documents.json"
        if bbc_path.exists():
            with open(bbc_path, 'r') as f:
                data_cache['bbc'] = json.load(f)
                logger.info(f"Loaded {len(data_cache['bbc'].get('documents', []))} BBC News documents")
        
        data_cache['last_loaded'] = datetime.now().isoformat()
        logger.info("All datasets loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        return False

def init_datasets():
    """Initialize datasets if they don't exist"""
    if not (processed_dir / "iris_vectors.json").exists() or \
       not (processed_dir / "wine_knowledge_graph.json").exists() or \
       not (processed_dir / "bbc_news_documents.json").exists():
        logger.info("Some datasets are missing, trying to run download_datasets.py")
        try:
            import subprocess
            result = subprocess.run(["python", "download_datasets.py"], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Successfully downloaded and processed datasets")
                return True
            else:
                logger.error(f"Error running download_datasets.py: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error initializing datasets: {e}")
            return False
    return True

# Dataset API routes
@dataset_bp.route('/info', methods=['GET'])
def dataset_info():
    """Get information about all datasets"""
    if data_cache['info'] is None:
        load_datasets()
    
    return jsonify(data_cache['info'])

@dataset_bp.route('/rzset/vectors', methods=['GET'])
def rzset_vectors():
    """Get vectors from the Iris dataset for the RZSet component"""
    if data_cache['iris'] is None:
        load_datasets()
    
    if data_cache['iris'] is None:
        return jsonify({"error": "Iris dataset not loaded"}), 404
    
    # Get query parameters
    limit = request.args.get('limit', default=100, type=int)
    group = request.args.get('group', default='iris', type=str)
    
    vectors = data_cache['iris'].get('vectors', [])
    
    # Sample vectors if limit is specified
    if limit < len(vectors):
        vectors = np.random.choice(vectors, limit, replace=False).tolist()
    
    return jsonify({
        "group": group,
        "vectors": vectors[:limit]
    })

@dataset_bp.route('/knowledge-graph/data', methods=['GET'])
def knowledge_graph_data():
    """Get data from the Wine dataset for the Knowledge Graph component"""
    if data_cache['wine'] is None:
        load_datasets()
    
    if data_cache['wine'] is None:
        return jsonify({"error": "Wine dataset not loaded"}), 404
    
    # Get query parameters
    limit = request.args.get('limit', default=100, type=int)
    
    nodes = data_cache['wine'].get('nodes', [])
    edges = data_cache['wine'].get('edges', [])
    
    # Limit nodes and edges if needed
    if limit < len(nodes):
        selected_nodes = nodes[:limit]
        selected_node_ids = {node['id'] for node in selected_nodes}
        
        # Only include edges that connect to selected nodes
        selected_edges = [
            edge for edge in edges 
            if edge['source'] in selected_node_ids and edge['target'] in selected_node_ids
        ]
    else:
        selected_nodes = nodes
        selected_edges = edges
    
    return jsonify({
        "nodes": selected_nodes,
        "edges": selected_edges
    })

@dataset_bp.route('/llm/documents', methods=['GET'])
def llm_documents():
    """Get documents from the BBC News dataset for the LLM component"""
    if data_cache['bbc'] is None:
        load_datasets()
    
    if data_cache['bbc'] is None:
        return jsonify({"error": "BBC News dataset not loaded"}), 404
    
    # Get query parameters
    limit = request.args.get('limit', default=10, type=int)
    category = request.args.get('category', default=None, type=str)
    query = request.args.get('query', default=None, type=str)
    
    documents = data_cache['bbc'].get('documents', [])
    
    # Filter by category if specified
    if category:
        documents = [doc for doc in documents if doc.get('category') == category]
    
    # Filter by search query if specified
    if query:
        query = query.lower()
        documents = [
            doc for doc in documents 
            if query in doc.get('title', '').lower() or query in doc.get('content', '').lower()
        ]
    
    # Get stats about the documents
    categories = {}
    for doc in data_cache['bbc'].get('documents', []):
        cat = doc.get('category')
        if cat:
            categories[cat] = categories.get(cat, 0) + 1
    
    # Sample documents if limit is specified
    if limit < len(documents):
        documents = np.random.choice(documents, limit, replace=False).tolist()
    
    return jsonify({
        "documents": documents[:limit],
        "stats": {
            "total": len(data_cache['bbc'].get('documents', [])),
            "filtered": len(documents),
            "categories": categories
        }
    })

@dataset_bp.route('/llm/document/<doc_id>', methods=['GET'])
def llm_document(doc_id):
    """Get a specific document from the BBC News dataset"""
    if data_cache['bbc'] is None:
        load_datasets()
    
    if data_cache['bbc'] is None:
        return jsonify({"error": "BBC News dataset not loaded"}), 404
    
    for doc in data_cache['bbc'].get('documents', []):
        if doc.get('id') == doc_id:
            return jsonify(doc)
    
    return jsonify({"error": f"Document with ID {doc_id} not found"}), 404

def register_with_darf_api(app):
    """Register dataset routes with DARF API server"""
    app.register_blueprint(dataset_bp, url_prefix='/api/datasets')
    logger.info("Registered dataset routes with DARF API server")
    
    # Initialize datasets
    init_datasets()
    
    # Preload datasets
    load_datasets()

# Can be used standalone for testing
if __name__ == "__main__":
    app = Flask(__name__)
    register_with_darf_api(app)
    
    print("Starting dataset connector standalone server...")
    port = 5050
    print(f"Running on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
