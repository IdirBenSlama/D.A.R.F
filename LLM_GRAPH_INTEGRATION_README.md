# DARF LLM-Knowledge Graph Integration

This integration connects LiteLLM with the DARF Knowledge Graph and 3D visualization, enabling AI-powered knowledge exploration, extraction, and analysis.

## Features

- **Natural Language Graph Queries**: Query the knowledge graph using conversational language
- **LLM-Enhanced Graph Visualization**: Interactive 3D visualization of knowledge entities with AI explanations
- **Fact Extraction**: Extract structured facts from natural language text using LLM
- **Node Analysis**: Get detailed AI-generated analysis of knowledge graph nodes
- **AI-Enhanced Exploration**: Discover new relationships and gain insights through AI analysis

## Components

1. **LLM Graph Interface** (`src/modules/llm_graph_interface/llm_graph_interface.py`)
   - Core integration between LiteLLM and knowledge graph
   - Handles vectorization, natural language queries, fact extraction

2. **API Endpoints** (in `darf_webapp/app.py`)
   - `/api/graph/query` - Natural language queries to the knowledge graph
   - `/api/graph/extract` - Extract facts from text
   - `/api/graph/enhance/<node_id>` - Get LLM-enhanced information about a node
   - `/api/vectorized` - Real knowledge graph nodes with vector representation

3. **JavaScript Integration** (`darf_webapp/static/js/llm_graph_integration.js`)
   - UI components for interacting with the LLM-Graph integration
   - Event handling for node selection and highlighting
   - Interactive exploration through query suggestions

4. **3D Visualization** (`darf_webapp/static/js/rzset.js`)
   - Real-time visualization of knowledge graph entities
   - Interactive node selection and exploration
   - Vector-based semantic positioning

## How to Use

### Starting the Integrated System

Run the integrated system using:

```bash
python run_darf_integrated.py
```

Optional flags:
- `--no-monitoring`: Skip starting the Prometheus/Grafana monitoring stack
- `--no-datasets`: Skip downloading and processing datasets

Then access the web interface at http://localhost:5000

### Natural Language Graph Queries

1. Navigate to the RZSet visualization page
2. Use the "Query Graph" button in the bottom-right toolbar
3. Enter a natural language query like "What connects to the knowledge graph?" or "Show all relationships between components"
4. View results in the dialog and see matching nodes highlighted in the visualization

### Exploring Nodes with LLM

1. Click on any node in the visualization
2. View LLM-enhanced information in the details panel
3. See related facts, AI analysis, and suggested queries
4. Click on suggested queries to explore further

### Extracting Facts from Text

1. Navigate to the Chat page and have a conversation
2. Return to the RZSet visualization page
3. Click "Extract Facts" in the toolbar
4. Facts will be extracted from your conversation and added to the knowledge graph
5. Refresh the visualization to see the new nodes

## Technical Details

- The integration uses AI-generated vector embeddings to position entities in 3D space
- LLM queries are translated into structured graph queries via prompt engineering
- Fact extraction uses structured output generation to parse SPO (Subject-Predicate-Object) triples
- API endpoints use asyncio to handle async LLM operations within Flask

## Integration with Monitoring

The system integrates with Prometheus and Grafana for monitoring:
- Prometheus endpoint: http://localhost:9090
- Grafana dashboard: http://localhost:3000 (credentials: admin/darf-admin)

## File Structure

```
src/modules/llm_graph_interface/
  ├── llm_graph_interface.py  # Core integration logic

darf_webapp/
  ├── app.py                  # API endpoints and web server
  ├── static/
  │   ├── css/
  │   │   └── llm_graph_integration.css  # Styling
  │   ├── js/
  │   │   ├── llm_graph_integration.js   # UI integration
  │   │   └── rzset.js                  # 3D visualization
  └── templates/
      └── rzset.html                    # Visualization page

run_darf_integrated.py       # Integrated runner
```

## Future Enhancements

- Semantic search across all knowledge graph nodes
- Automated hypothesis generation from facts
- Multi-modal knowledge representation (text, images, code)
- Real-time collaborative exploration
- Temporal analysis of evolving knowledge
