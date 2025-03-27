#!/usr/bin/env python
from flask import Flask, render_template_string, jsonify
import random
import time
import datetime

app = Flask(__name__)

# Main template with navigation and dynamic content loading
MAIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DARF Framework - {{ page_title }}</title>
    <style>
        :root {
            --primary-color: #3498db;
            --secondary-color: #2ecc71;
            --dark-color: #2c3e50;
            --light-color: #ecf0f1;
            --warning-color: #e74c3c;
            --info-color: #f1c40f;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f7fa;
            color: #333;
        }
        
        header {
            background-color: var(--dark-color);
            color: white;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .container {
            display: flex;
            min-height: calc(100vh - 60px);
        }
        
        nav {
            width: 200px;
            background-color: #2c3e50;
            color: white;
            padding: 1rem 0;
        }
        
        nav ul {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        
        nav li {
            margin-bottom: 0.5rem;
        }
        
        nav a {
            display: block;
            padding: 0.5rem 1rem;
            color: white;
            text-decoration: none;
            transition: background-color 0.3s;
        }
        
        nav a:hover, nav a.active {
            background-color: #34495e;
            border-left: 4px solid var(--primary-color);
        }
        
        main {
            flex: 1;
            padding: 1.5rem;
        }
        
        .card {
            background-color: white;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        
        .card-header {
            border-bottom: 1px solid #eee;
            padding-bottom: 0.5rem;
            margin-bottom: 1rem;
            font-weight: bold;
            font-size: 1.2rem;
        }
        
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 0.5rem;
        }
        
        .status-active {
            background-color: var(--secondary-color);
        }
        
        .status-warning {
            background-color: var(--info-color);
        }
        
        .status-error {
            background-color: var(--warning-color);
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 1rem;
        }
        
        .metric-card {
            background-color: white;
            border-radius: 4px;
            padding: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: var(--primary-color);
        }
        
        .metric-label {
            color: #7f8c8d;
            font-size: 0.9rem;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }
        
        table th, table td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        
        table th {
            background-color: #f8f9fa;
        }
        
        .refresh-btn {
            background-color: var(--primary-color);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            cursor: pointer;
            float: right;
        }
    </style>
</head>
<body>
    <header>
        <h1>DARF Framework</h1>
    </header>
    <div class="container">
        <nav>
            <ul>
                <li><a href="/" {% if active_page == 'dashboard' %}class="active"{% endif %}>Dashboard</a></li>
                <li><a href="/knowledge" {% if active_page == 'knowledge' %}class="active"{% endif %}>Knowledge Graph</a></li>
                <li><a href="/events" {% if active_page == 'events' %}class="active"{% endif %}>Event Bus</a></li>
                <li><a href="/rzset" {% if active_page == 'rzset' %}class="active"{% endif %}>RZ Set</a></li>
                <li><a href="/llm" {% if active_page == 'llm' %}class="active"{% endif %}>LLM Interface</a></li>
                <li><a href="/metrics" {% if active_page == 'metrics' %}class="active"{% endif %}>Metrics</a></li>
                <li><a href="/consensus" {% if active_page == 'consensus' %}class="active"{% endif %}>Consensus</a></li>
                <li><a href="/settings" {% if active_page == 'settings' %}class="active"{% endif %}>Settings</a></li>
            </ul>
        </nav>
        <main>
            {% block content %}{% endblock %}
        </main>
    </div>
    <script>
        // Common JavaScript functions
        function refreshData() {
            location.reload();
        }
    </script>
</body>
</html>
"""

# Dashboard template
DASHBOARD_TEMPLATE = """
{% extends base_template %}
{% block content %}
    <h2>Dashboard</h2>
    <p>Last Updated: {{ current_time }}</p>
    <button class="refresh-btn" onclick="refreshData()">Refresh</button>
    
    <div class="card">
        <div class="card-header">System Status</div>
        <div>
            <p><span class="status-indicator status-active"></span> <strong>Status:</strong> Running</p>
            <p><strong>Mode:</strong> Standard</p>
            <p><strong>Uptime:</strong> {{ uptime }}</p>
        </div>
    </div>
    
    <div class="grid">
        <div class="metric-card">
            <div class="metric-label">Knowledge Graph Entities</div>
            <div class="metric-value">{{ metrics.knowledge_graph.entity_count }}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Event Bus Messages</div>
            <div class="metric-value">{{ metrics.event_bus.event_count }}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">CPU Usage</div>
            <div class="metric-value">{{ metrics.system.cpu }}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Memory Usage</div>
            <div class="metric-value">{{ metrics.system.memory }}%</div>
        </div>
    </div>
    
    <div class="card">
        <div class="card-header">Component Status</div>
        <table>
            <thead>
                <tr>
                    <th>Component</th>
                    <th>Status</th>
                    <th>Details</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Knowledge Graph</td>
                    <td><span class="status-indicator status-active"></span> Active</td>
                    <td>{{ metrics.knowledge_graph.entity_count }} entities, {{ metrics.knowledge_graph.relation_count }} relations</td>
                </tr>
                <tr>
                    <td>Event Bus</td>
                    <td><span class="status-indicator status-active"></span> Running</td>
                    <td>{{ metrics.event_bus.events_per_second }} events/sec, Queue: {{ metrics.event_bus.queue_size }}</td>
                </tr>
                <tr>
                    <td>LLM Interface</td>
                    <td><span class="status-indicator status-active"></span> Connected</td>
                    <td>Default model, Embeddings initialized</td>
                </tr>
                <tr>
                    <td>Web Interface</td>
                    <td><span class="status-indicator status-active"></span> Running</td>
                    <td>Port 5000, Debug mode</td>
                </tr>
            </tbody>
        </table>
    </div>
    
    <div class="card">
        <div class="card-header">Recent Activities</div>
        <table>
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Component</th>
                    <th>Activity</th>
                </tr>
            </thead>
            <tbody>
                {% for activity in activities %}
                <tr>
                    <td>{{ activity.time }}</td>
                    <td>{{ activity.component }}</td>
                    <td>{{ activity.message }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
{% endblock %}
"""

# Knowledge Graph template
KNOWLEDGE_GRAPH_TEMPLATE = """
{% extends base_template %}
{% block content %}
    <h2>Knowledge Graph</h2>
    <p>Last Updated: {{ current_time }}</p>
    <button class="refresh-btn" onclick="refreshData()">Refresh</button>
    
    <div class="card">
        <div class="card-header">Graph Summary</div>
        <div class="grid">
            <div class="metric-card">
                <div class="metric-label">Entities</div>
                <div class="metric-value">{{ metrics.knowledge_graph.entity_count }}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Relations</div>
                <div class="metric-value">{{ metrics.knowledge_graph.relation_count }}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Facts</div>
                <div class="metric-value">{{ metrics.knowledge_graph.fact_count }}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Queries</div>
                <div class="metric-value">{{ metrics.knowledge_graph.query_count }}</div>
            </div>
        </div>
    </div>
    
    <div class="card">
        <div class="card-header">Recent Facts</div>
        <table>
            <thead>
                <tr>
                    <th>Subject</th>
                    <th>Predicate</th>
                    <th>Object</th>
                    <th>Confidence</th>
                </tr>
            </thead>
            <tbody>
                {% for fact in facts %}
                <tr>
                    <td>{{ fact.subject }}</td>
                    <td>{{ fact.predicate }}</td>
                    <td>{{ fact.object }}</td>
                    <td>{{ fact.confidence }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
{% endblock %}
"""

# Event Bus template
EVENTS_TEMPLATE = """
{% extends base_template %}
{% block content %}
    <h2>Event Bus</h2>
    <p>Last Updated: {{ current_time }}</p>
    <button class="refresh-btn" onclick="refreshData()">Refresh</button>
    
    <div class="card">
        <div class="card-header">Event Bus Statistics</div>
        <div class="grid">
            <div class="metric-card">
                <div class="metric-label">Total Events</div>
                <div class="metric-value">{{ metrics.event_bus.event_count }}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Events Per Second</div>
                <div class="metric-value">{{ metrics.event_bus.events_per_second }}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Current Queue Size</div>
                <div class="metric-value">{{ metrics.event_bus.queue_size }}</div>
            </div>
        </div>
    </div>
    
    <div class="card">
        <div class="card-header">Recent Events</div>
        <table>
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Event Type</th>
                    <th>Source</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {% for event in events %}
                <tr>
                    <td>{{ event.timestamp }}</td>
                    <td>{{ event.type }}</td>
                    <td>{{ event.source }}</td>
                    <td>{{ event.status }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
{% endblock %}
"""

# Metrics template 
METRICS_TEMPLATE = """
{% extends base_template %}
{% block content %}
    <h2>System Metrics</h2>
    <p>Last Updated: {{ current_time }}</p>
    <button class="refresh-btn" onclick="refreshData()">Refresh</button>
    
    <div class="card">
        <div class="card-header">System Resources</div>
        <div class="grid">
            <div class="metric-card">
                <div class="metric-label">CPU Usage</div>
                <div class="metric-value">{{ metrics.system.cpu }}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Memory Usage</div>
                <div class="metric-value">{{ metrics.system.memory }}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Disk Usage</div>
                <div class="metric-value">{{ metrics.system.disk }}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Uptime (seconds)</div>
                <div class="metric-value">{{ metrics.system.uptime }}</div>
            </div>
        </div>
    </div>
    
    <div class="card">
        <div class="card-header">Dataset Statistics</div>
        <div class="grid">
            <div class="metric-card">
                <div class="metric-label">Dataset Count</div>
                <div class="metric-value">{{ metrics.datasets.count }}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Rows</div>
                <div class="metric-value">{{ metrics.datasets.total_rows }}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Active Connections</div>
                <div class="metric-value">{{ metrics.datasets.active_connections }}</div>
            </div>
        </div>
    </div>
{% endblock %}
"""

# Common function to generate random metrics
def get_metrics():
    return {
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
    }

# Generate random activities
def get_activities(count=10):
    components = ['Knowledge Graph', 'Event Bus', 'LLM Interface', 'System Monitor', 'Dataset Connector']
    actions = [
        'Entity added', 'Relation updated', 'Query executed', 'Event processed', 
        'Model loaded', 'Metrics collected', 'Dataset connection established',
        'Cache updated', 'Index optimized', 'Configuration reloaded'
    ]
    
    activities = []
    now = datetime.datetime.now()
    
    for i in range(count):
        time_ago = now - datetime.timedelta(minutes=random.randint(1, 60))
        activities.append({
            'time': time_ago.strftime("%H:%M:%S"),
            'component': random.choice(components),
            'message': random.choice(actions)
        })
    
    return sorted(activities, key=lambda x: x['time'], reverse=True)

# Generate random facts
def get_facts(count=8):
    entities = ['DARF', 'Knowledge Graph', 'RZSet', 'Vector', 'Consensus',
                'LLM', 'Database', 'Python', 'Neural Network', 'Dataset']
    predicates = ['connects_to', 'is_part_of', 'uses', 'depends_on', 
                 'processes', 'analyzes', 'generates', 'trains']
    
    facts = []
    for _ in range(count):
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
    
    return facts

# Generate random events
def get_events(count=8):
    event_types = ['EntityCreated', 'RelationAdded', 'QueryExecuted', 'ModelLoaded',
                  'CacheUpdated', 'DatasetProcessed', 'ConfigChanged', 'MetricsCollected']
    sources = ['KnowledgeGraph', 'EventBus', 'LLMInterface', 'DataConnector', 'SystemMonitor']
    statuses = ['Completed', 'Processing', 'Completed', 'Completed', 'Queued']
    
    events = []
    now = time.time()
    
    for i in range(count):
        timestamp = now - random.randint(1, 3600)
        events.append({
            'timestamp': datetime.datetime.fromtimestamp(timestamp).strftime("%H:%M:%S"),
            'type': random.choice(event_types),
            'source': random.choice(sources),
            'status': random.choice(statuses)
        })
    
    return sorted(events, key=lambda x: x['timestamp'], reverse=True)

@app.route('/')
def dashboard():
    metrics = get_metrics()
    activities = get_activities()
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    uptime = "1 day, 3 hours, 24 minutes"
    
    return render_template_string(
        DASHBOARD_TEMPLATE,
        base_template=MAIN_TEMPLATE,
        active_page='dashboard',
        page_title='Dashboard',
        metrics=metrics,
        activities=activities,
        current_time=current_time,
        uptime=uptime
    )

@app.route('/knowledge')
def knowledge():
    metrics = get_metrics()
    facts = get_facts()
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return render_template_string(
        KNOWLEDGE_GRAPH_TEMPLATE,
        base_template=MAIN_TEMPLATE,
        active_page='knowledge',
        page_title='Knowledge Graph',
        metrics=metrics,
        facts=facts,
        current_time=current_time
    )

@app.route('/events')
def events():
    metrics = get_metrics()
    events_data = get_events()
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return render_template_string(
        EVENTS_TEMPLATE,
        base_template=MAIN_TEMPLATE,
        active_page='events',
        page_title='Event Bus',
        metrics=metrics,
        events=events_data,
        current_time=current_time
    )

@app.route('/metrics')
def metrics():
    metrics_data = get_metrics()
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return render_template_string(
        METRICS_TEMPLATE,
        base_template=MAIN_TEMPLATE,
        active_page='metrics',
        page_title='System Metrics',
        metrics=metrics_data,
        current_time=current_time
    )

# Placeholder routes for other sections (simplified)
@app.route('/rzset')
def rzset():
    return render_template_string(
        "{% extends base_template %}{% block content %}<h2>RZ Set Visualization</h2><p>This feature is currently in development.</p>{% endblock %}",
        base_template=MAIN_TEMPLATE,
        active_page='rzset',
        page_title='RZ Set'
    )

@app.route('/llm')
def llm():
    return render_template_string(
        "{% extends base_template %}{% block content %}<h2>LLM Interface</h2><p>This feature is currently in development.</p>{% endblock %}",
        base_template=MAIN_TEMPLATE,
        active_page='llm',
        page_title='LLM Interface'
    )

@app.route('/consensus')
def consensus():
    return render_template_string(
        "{% extends base_template %}{% block content %}<h2>Consensus Module</h2><p>This feature is currently in development.</p>{% endblock %}",
        base_template=MAIN_TEMPLATE,
        active_page='consensus',
        page_title='Consensus'
    )

@app.route('/settings')
def settings():
    return render_template_string(
        "{% extends base_template %}{% block content %}<h2>Settings</h2><p>This feature is currently in development.</p>{% endblock %}",
        base_template=MAIN_TEMPLATE,
        active_page='settings',
        page_title='Settings'
    )

# API endpoints for getting data
@app.route('/api/metrics')
def api_metrics():
    return jsonify(get_metrics())

@app.route('/api/facts')
def api_facts():
    return jsonify({
        'facts': get_facts(),
        'count': 8,
        'timestamp': time.time()
    })

@app.route('/api/events')
def api_events():
    return jsonify({
        'events': get_events(),
        'count': 8,
        'timestamp': time.time()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
