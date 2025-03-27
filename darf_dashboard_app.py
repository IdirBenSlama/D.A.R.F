#!/usr/bin/env python3
"""
DARF Interactive Dashboard
Standalone version for quick testing
"""

import os
import random
import time
import json
from datetime import datetime
from flask import Flask, render_template, jsonify, request, redirect, url_for, send_from_directory

app = Flask(__name__, 
            static_folder='darf_webapp/static',
            template_folder='darf_webapp/templates')

# Default configuration
app.config.update(
    SECRET_KEY='dev-key-for-darf-dashboard',
    DEBUG=True,
    TESTING=False,
)

# API routes for the dashboard
@app.route('/api/metrics')
def api_metrics():
    """Return system metrics."""
    # Generate random metrics for demonstration
    return jsonify({
        'knowledge_graph': {
            'entity_count': random.randint(500, 2000),
            'relation_count': random.randint(1000, 5000),
            'fact_count': random.randint(2000, 10000),
            'query_count': random.randint(100, 1000)
        },
        'event_bus': {
            'event_count': random.randint(1000, 10000),
            'events_per_second': random.randint(10, 100),
            'queue_size': random.randint(0, 50)
        },
        'datasets': {
            'count': random.randint(5, 20),
            'total_rows': random.randint(10000, 1000000),
            'active_connections': random.randint(1, 10)
        },
        'system': {
            'cpu': random.randint(10, 90),
            'memory': random.randint(20, 80),
            'disk': random.randint(30, 70),
            'uptime': random.randint(100, 10000)
        },
        'timestamp': time.time()
    })

@app.route('/api/facts')
def api_facts():
    """Return knowledge graph facts."""
    limit = request.args.get('limit', 20, type=int)
    
    # Sample entities and predicates
    entities = ['DARF', 'Knowledge Graph', 'RZSet', 'Vector', 'Consensus',
                'LLM', 'Database', 'Python', 'Neural Network', 'Dataset']
    
    predicates = ['connects_to', 'is_part_of', 'uses', 'depends_on', 
                 'processes', 'analyzes', 'generates', 'trains']
    
    # Generate random facts
    facts = []
    for _ in range(limit):
        subject = random.choice(entities)
        predicate = random.choice(predicates)
        object_ = random.choice([e for e in entities if e != subject])
        confidence = round(random.uniform(0.5, 1.0), 2)
        
        facts.append({
            'subject': subject,
            'predicate': predicate,
            'object': object_,
            'confidence': confidence
        })
    
    return jsonify({
        'facts': facts,
        'count': len(facts),
        'timestamp': time.time()
    })

@app.route('/api/consensus/status')
def api_consensus_status():
    """Return consensus module status."""
    nodes = random.randint(3, 10)
    return jsonify({
        'nodes': nodes,
        'agreement_rate': round(random.uniform(0.7, 1.0), 2),
        'transactions_per_second': random.randint(10, 1000),
        'leader': f'Node-{random.randint(1, nodes)}',
        'epoch': random.randint(1, 100),
        'last_commit': int(time.time()) - random.randint(1, 60),
        'status': random.choice(['healthy', 'degraded', 'recovering', 'healthy', 'healthy'])
    })

@app.route('/api/vectorized')
def api_vectorized():
    """Return vectorized data for RZSet visualization."""
    # Generate random 2D vectors for visualization
    count = random.randint(20, 50)
    vectors = []
    
    groups = ['group1', 'group2', 'group3', 'group4']
    
    for i in range(count):
        # Generate random 2D coordinates between -1 and 1
        vectors.append({
            'id': f'vec_{i}',
            'x': random.uniform(-1, 1),
            'y': random.uniform(-1, 1),
            'group': random.choice(groups),
            'size': random.uniform(0.2, 1.0),
            'label': f'V{i}' if random.random() > 0.7 else None
        })
    
    return jsonify({
        'vectors': vectors,
        'count': len(vectors),
        'dimensions': 2,
        'timestamp': time.time()
    })

@app.route('/api/graph/query', methods=['POST'])
def api_graph_query():
    """Handle LLM queries to the knowledge graph."""
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    # Simulate processing time
    time.sleep(random.uniform(0.5, 2.0))
    
    # Generate a response based on the query
    relevant_terms = ['knowledge', 'graph', 'data', 'consensus', 'llm', 'neural', 
                      'vector', 'database', 'rzset', 'darf']
    
    # Check if query contains any of the relevant terms
    has_relevant_terms = any(term in query.lower() for term in relevant_terms)
    
    if has_relevant_terms:
        # Generate random facts as results
        results = []
        for _ in range(random.randint(1, 5)):
            results.append({
                'subject': random.choice(['DARF', 'RZSet', 'Knowledge Graph']),
                'predicate': random.choice(['contains', 'processes', 'analyzes']),
                'object': random.choice(['data', 'vectors', 'entities']),
                'confidence': random.uniform(0.7, 0.95)
            })
        
        summary = f"Found {len(results)} results related to your query. The DARF system shows connections between the requested concepts."
        
        return jsonify({
            'results': results,
            'summary': summary,
            'success': True
        })
    else:
        # No relevant results
        return jsonify({
            'results': [],
            'summary': "No relevant information found in the knowledge graph for this query.",
            'success': True
        })

@app.route('/api/datasets')
def api_datasets():
    """Return available datasets."""
    datasets = [
        {'id': 'covid', 'name': 'COVID-19 Dataset', 'size': '1.2 GB', 'rows': 125000, 'format': 'CSV'},
        {'id': 'finance', 'name': 'Financial Data', 'size': '890 MB', 'rows': 75000, 'format': 'JSON'},
        {'id': 'wiki', 'name': 'Wikipedia Dump', 'size': '5.4 GB', 'rows': 1200000, 'format': 'XML'},
    ]
    
    return jsonify({
        'datasets': datasets,
        'count': len(datasets),
        'timestamp': time.time()
    })

@app.route('/api/databases')
def api_databases():
    """Return available databases."""
    databases = [
        {'id': 'postgres', 'name': 'PostgreSQL', 'type': 'SQL', 'status': 'available'},
        {'id': 'mongo', 'name': 'MongoDB', 'type': 'NoSQL', 'status': 'available'},
        {'id': 'neo4j', 'name': 'Neo4j', 'type': 'Graph', 'status': 'available'},
    ]
    
    return jsonify({
        'databases': databases,
        'count': len(databases),
        'timestamp': time.time()
    })

@app.route('/api/exports')
def api_exports():
    """Return recent exports."""
    now = int(time.time())
    exports = [
        {'id': 'export1', 'date': now - 3600, 'format': 'JSON', 'size': '1.2 MB', 'status': 'completed'},
        {'id': 'export2', 'date': now - 86400, 'format': 'CSV', 'size': '890 KB', 'status': 'completed'},
    ]
    
    return jsonify({
        'exports': exports,
        'count': len(exports),
        'timestamp': time.time()
    })

@app.route('/favicon.ico')
def favicon():
    """Serve the favicon."""
    return send_from_directory(os.path.join(app.root_path, 'darf_webapp/static'), 
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

# Main routes
@app.route('/')
def index():
    """Render main dashboard page."""
    return render_template('enhanced_dashboard.html')

@app.route('/dashboard')
def dashboard():
    """Redirect to the main dashboard."""
    return redirect(url_for('index'))

@app.route('/classic')
def classic():
    """Render classic dashboard view."""
    return render_template('index.html')

@app.route('/knowledge')
def knowledge():
    """Render knowledge graph page."""
    return render_template('knowledge.html')

@app.route('/rzset')
def rzset():
    """Render RZSet visualization page."""
    return render_template('rzset.html')

@app.route('/llm')
def llm():
    """Render LLM interface page."""
    return render_template('llm.html')

@app.route('/consensus')
def consensus():
    """Render consensus page."""
    return render_template('consensus.html')

@app.route('/resources')
def resources():
    """Render resources page."""
    return render_template('resources.html')

@app.route('/metrics')
def metrics():
    """Render metrics page."""
    return render_template('metrics.html')

@app.route('/settings')
def settings():
    """Render settings page."""
    return render_template('settings.html')

@app.route('/events')
def events():
    """Render events page."""
    return render_template('events.html')

@app.route('/graph')
def graph():
    """Render graph page."""
    return render_template('graph.html')

@app.route('/click-here')
def click_here():
    """Handle the 'click here' link on the main dashboard."""
    return render_template('unified.html')

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', 
                           error_code=404, 
                           error_message="The page you requested was not found."), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html', 
                           error_code=500, 
                           error_message="An internal server error occurred."), 500

if __name__ == '__main__':
    print("DARF Dashboard running at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
