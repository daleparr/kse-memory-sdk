"""
KSE Memory SDK - Visual Dashboard

Web-based dashboard for exploring and understanding
hybrid AI search capabilities with interactive visualizations.
"""

import asyncio
import json
import webbrowser
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from ..core.memory import KSEMemory
from ..core.models import Product, SearchQuery, SearchType
from .conceptual_explorer import ConceptualSpaceExplorer
from .graph_visualizer import KnowledgeGraphVisualizer
from .search_explainer import SearchResultsExplainer


class KSEDashboard:
    """
    Interactive web dashboard for KSE Memory exploration.
    
    Provides real-time visualization of:
    - Conceptual space in 3D
    - Knowledge graph relationships
    - Search result explanations
    - Performance metrics
    
    Example:
        dashboard = KSEDashboard(kse_memory)
        await dashboard.start(port=8080, open_browser=True)
    """
    
    def __init__(self, kse_memory: KSEMemory, host: str = "localhost", port: int = 8080):
        """
        Initialize KSE Dashboard.
        
        Args:
            kse_memory: KSE Memory instance to visualize
            host: Host to bind the server to
            port: Port to run the server on
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError(
                "FastAPI is required for the visual dashboard. "
                "Install with: pip install fastapi uvicorn"
            )
        
        self.kse_memory = kse_memory
        self.host = host
        self.port = port
        
        # Initialize visualization components
        self.conceptual_explorer = ConceptualSpaceExplorer(kse_memory)
        self.graph_visualizer = KnowledgeGraphVisualizer(kse_memory)
        self.search_explainer = SearchResultsExplainer(kse_memory)
        
        # FastAPI app
        self.app = FastAPI(title="KSE Memory Dashboard", version="1.0.0")
        self._setup_routes()
        
        # WebSocket connections for real-time updates
        self.active_connections: List[WebSocket] = []
    
    def _setup_routes(self):
        """Setup FastAPI routes for the dashboard."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home():
            """Serve the main dashboard page."""
            return self._generate_dashboard_html()
        
        @self.app.get("/api/conceptual-space")
        async def get_conceptual_space():
            """Get conceptual space data for visualization."""
            return await self.conceptual_explorer.get_space_data()
        
        @self.app.get("/api/knowledge-graph")
        async def get_knowledge_graph():
            """Get knowledge graph data for visualization."""
            return await self.graph_visualizer.get_graph_data()
        
        @self.app.post("/api/search")
        async def search_products(request: Dict[str, Any]):
            """Perform search and return explained results."""
            query = request.get("query", "")
            search_type = request.get("search_type", "hybrid")
            limit = request.get("limit", 10)
            
            # Perform search
            search_query = SearchQuery(
                query=query,
                search_type=SearchType(search_type.upper()),
                limit=limit
            )
            
            results = await self.kse_memory.search(search_query)
            
            # Get explanation
            explanation = await self.search_explainer.explain_results(
                query, results, search_type
            )
            
            # Broadcast to connected clients
            await self._broadcast_search_update({
                "query": query,
                "results": [r.to_dict() for r in results],
                "explanation": explanation,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return {
                "results": [r.to_dict() for r in results],
                "explanation": explanation
            }
        
        @self.app.get("/api/performance")
        async def get_performance_metrics():
            """Get current performance metrics."""
            return await self._get_performance_metrics()
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await websocket.accept()
            self.active_connections.append(websocket)
            
            try:
                while True:
                    # Keep connection alive and handle incoming messages
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    if message.get("type") == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))
                    
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
    
    async def _broadcast_search_update(self, data: Dict[str, Any]):
        """Broadcast search updates to all connected clients."""
        message = json.dumps({
            "type": "search_update",
            "data": data
        })
        
        # Remove disconnected clients
        active_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
                active_connections.append(connection)
            except:
                pass  # Connection closed
        
        self.active_connections = active_connections
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        # In a real implementation, this would collect actual metrics
        return {
            "search_latency_ms": 45.2,
            "memory_usage_mb": 128.5,
            "active_connections": len(self.active_connections),
            "total_products": await self._get_product_count(),
            "cache_hit_rate": 0.85,
            "queries_per_second": 12.3,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _get_product_count(self) -> int:
        """Get total number of products in memory."""
        # This would query the actual KSE Memory instance
        return 1000  # Placeholder
    
    def _generate_dashboard_html(self) -> str:
        """Generate the main dashboard HTML page."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KSE Memory Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.21.1/cytoscape.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f1419;
            color: #ffffff;
            overflow: hidden;
        }
        
        .dashboard-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 60px 1fr 1fr;
            height: 100vh;
            gap: 1px;
            background: #1a1a1a;
        }
        
        .header {
            grid-column: 1 / -1;
            background: #2d3748;
            display: flex;
            align-items: center;
            padding: 0 20px;
            border-bottom: 1px solid #4a5568;
        }
        
        .header h1 {
            color: #63b3ed;
            font-size: 24px;
            font-weight: 600;
        }
        
        .search-container {
            margin-left: auto;
            display: flex;
            gap: 10px;
        }
        
        .search-input {
            padding: 8px 12px;
            border: 1px solid #4a5568;
            border-radius: 6px;
            background: #1a202c;
            color: #ffffff;
            width: 300px;
        }
        
        .search-button {
            padding: 8px 16px;
            background: #3182ce;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        .search-button:hover {
            background: #2c5aa0;
        }
        
        .panel {
            background: #1a202c;
            border: 1px solid #2d3748;
            position: relative;
            overflow: hidden;
        }
        
        .panel-header {
            background: #2d3748;
            padding: 12px 16px;
            border-bottom: 1px solid #4a5568;
            font-weight: 600;
            font-size: 14px;
            color: #a0aec0;
        }
        
        .panel-content {
            height: calc(100% - 49px);
            padding: 16px;
            overflow: auto;
        }
        
        #conceptual-space {
            width: 100%;
            height: 100%;
        }
        
        #knowledge-graph {
            width: 100%;
            height: 100%;
        }
        
        .search-results {
            max-height: 100%;
            overflow-y: auto;
        }
        
        .result-item {
            background: #2d3748;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 12px;
            border-left: 4px solid #3182ce;
        }
        
        .result-title {
            font-weight: 600;
            color: #63b3ed;
            margin-bottom: 4px;
        }
        
        .result-score {
            font-size: 12px;
            color: #68d391;
            margin-bottom: 8px;
        }
        
        .result-description {
            font-size: 14px;
            color: #a0aec0;
            line-height: 1.4;
        }
        
        .performance-metrics {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }
        
        .metric-card {
            background: #2d3748;
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: 700;
            color: #63b3ed;
            margin-bottom: 4px;
        }
        
        .metric-label {
            font-size: 12px;
            color: #a0aec0;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #68d391;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #a0aec0;
        }
    </style>
</head>
<body>
    <div class="dashboard-container">
        <div class="header">
            <h1>KSE Memory Dashboard</h1>
            <div class="search-container">
                <input type="text" class="search-input" id="searchInput" placeholder="Search products..." />
                <button class="search-button" onclick="performSearch()">Search</button>
            </div>
        </div>
        
        <div class="panel">
            <div class="panel-header">
                <span class="status-indicator"></span>
                Conceptual Space Explorer
            </div>
            <div class="panel-content">
                <div id="conceptual-space" class="loading">
                    Loading 3D conceptual space...
                </div>
            </div>
        </div>
        
        <div class="panel">
            <div class="panel-header">
                <span class="status-indicator"></span>
                Knowledge Graph
            </div>
            <div class="panel-content">
                <div id="knowledge-graph" class="loading">
                    Loading knowledge graph...
                </div>
            </div>
        </div>
        
        <div class="panel">
            <div class="panel-header">
                <span class="status-indicator"></span>
                Search Results & Explanation
            </div>
            <div class="panel-content">
                <div id="search-results" class="search-results">
                    <div style="text-align: center; color: #a0aec0; margin-top: 50px;">
                        Enter a search query to see hybrid AI results
                    </div>
                </div>
            </div>
        </div>
        
        <div class="panel">
            <div class="panel-header">
                <span class="status-indicator"></span>
                Performance Metrics
            </div>
            <div class="panel-content">
                <div id="performance-metrics" class="performance-metrics">
                    <div class="metric-card">
                        <div class="metric-value" id="latency">--</div>
                        <div class="metric-label">Avg Latency (ms)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="memory">--</div>
                        <div class="metric-label">Memory (MB)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="products">--</div>
                        <div class="metric-label">Total Products</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="qps">--</div>
                        <div class="metric-label">Queries/sec</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // WebSocket connection for real-time updates
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        ws.onmessage = function(event) {
            const message = JSON.parse(event.data);
            if (message.type === 'search_update') {
                updateSearchResults(message.data);
            }
        };
        
        // Initialize dashboard
        async function initDashboard() {
            await loadConceptualSpace();
            await loadKnowledgeGraph();
            await loadPerformanceMetrics();
            
            // Update metrics every 5 seconds
            setInterval(loadPerformanceMetrics, 5000);
        }
        
        async function loadConceptualSpace() {
            try {
                const response = await fetch('/api/conceptual-space');
                const data = await response.json();
                renderConceptualSpace(data);
            } catch (error) {
                console.error('Failed to load conceptual space:', error);
                document.getElementById('conceptual-space').innerHTML = 
                    '<div style="color: #f56565;">Failed to load conceptual space</div>';
            }
        }
        
        async function loadKnowledgeGraph() {
            try {
                const response = await fetch('/api/knowledge-graph');
                const data = await response.json();
                renderKnowledgeGraph(data);
            } catch (error) {
                console.error('Failed to load knowledge graph:', error);
                document.getElementById('knowledge-graph').innerHTML = 
                    '<div style="color: #f56565;">Failed to load knowledge graph</div>';
            }
        }
        
        async function loadPerformanceMetrics() {
            try {
                const response = await fetch('/api/performance');
                const data = await response.json();
                updatePerformanceMetrics(data);
            } catch (error) {
                console.error('Failed to load performance metrics:', error);
            }
        }
        
        function renderConceptualSpace(data) {
            const container = document.getElementById('conceptual-space');
            container.innerHTML = '';
            
            // Create Three.js scene
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
            const renderer = new THREE.WebGLRenderer({ antialias: true });
            
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.setClearColor(0x0f1419);
            container.appendChild(renderer.domElement);
            
            // Add sample points for demonstration
            const geometry = new THREE.SphereGeometry(0.05, 16, 16);
            const material = new THREE.MeshBasicMaterial({ color: 0x63b3ed });
            
            for (let i = 0; i < 50; i++) {
                const sphere = new THREE.Mesh(geometry, material);
                sphere.position.set(
                    (Math.random() - 0.5) * 4,
                    (Math.random() - 0.5) * 4,
                    (Math.random() - 0.5) * 4
                );
                scene.add(sphere);
            }
            
            // Add axes
            const axesHelper = new THREE.AxesHelper(2);
            scene.add(axesHelper);
            
            camera.position.z = 5;
            
            // Animation loop
            function animate() {
                requestAnimationFrame(animate);
                scene.rotation.y += 0.005;
                renderer.render(scene, camera);
            }
            animate();
        }
        
        function renderKnowledgeGraph(data) {
            const container = document.getElementById('knowledge-graph');
            container.innerHTML = '';
            
            // Create Cytoscape graph
            const cy = cytoscape({
                container: container,
                style: [
                    {
                        selector: 'node',
                        style: {
                            'background-color': '#63b3ed',
                            'label': 'data(label)',
                            'color': '#ffffff',
                            'font-size': '12px',
                            'text-valign': 'center',
                            'text-halign': 'center',
                            'width': '30px',
                            'height': '30px'
                        }
                    },
                    {
                        selector: 'edge',
                        style: {
                            'width': 2,
                            'line-color': '#4a5568',
                            'target-arrow-color': '#4a5568',
                            'target-arrow-shape': 'triangle'
                        }
                    }
                ],
                elements: [
                    // Sample nodes and edges
                    { data: { id: 'shoes', label: 'Shoes' } },
                    { data: { id: 'athletic', label: 'Athletic' } },
                    { data: { id: 'formal', label: 'Formal' } },
                    { data: { id: 'comfort', label: 'Comfort' } },
                    { data: { source: 'shoes', target: 'athletic' } },
                    { data: { source: 'shoes', target: 'formal' } },
                    { data: { source: 'athletic', target: 'comfort' } }
                ],
                layout: {
                    name: 'cose',
                    animate: true,
                    animationDuration: 1000
                }
            });
        }
        
        function updatePerformanceMetrics(data) {
            document.getElementById('latency').textContent = data.search_latency_ms.toFixed(1);
            document.getElementById('memory').textContent = data.memory_usage_mb.toFixed(1);
            document.getElementById('products').textContent = data.total_products.toLocaleString();
            document.getElementById('qps').textContent = data.queries_per_second.toFixed(1);
        }
        
        async function performSearch() {
            const query = document.getElementById('searchInput').value;
            if (!query.trim()) return;
            
            try {
                const response = await fetch('/api/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        query: query,
                        search_type: 'hybrid',
                        limit: 5
                    })
                });
                
                const data = await response.json();
                updateSearchResults({ query, ...data });
            } catch (error) {
                console.error('Search failed:', error);
            }
        }
        
        function updateSearchResults(data) {
            const container = document.getElementById('search-results');
            
            if (!data.results || data.results.length === 0) {
                container.innerHTML = '<div style="text-align: center; color: #a0aec0;">No results found</div>';
                return;
            }
            
            let html = `<div style="margin-bottom: 16px; color: #63b3ed; font-weight: 600;">
                Results for "${data.query}" (${data.results.length} found)
            </div>`;
            
            data.results.forEach((result, index) => {
                html += `
                    <div class="result-item">
                        <div class="result-title">${result.product.title}</div>
                        <div class="result-score">Score: ${result.score.toFixed(3)} | Rank: ${index + 1}</div>
                        <div class="result-description">${result.product.description}</div>
                    </div>
                `;
            });
            
            container.innerHTML = html;
        }
        
        // Handle Enter key in search input
        document.getElementById('searchInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                performSearch();
            }
        });
        
        // Initialize dashboard when page loads
        window.addEventListener('load', initDashboard);
        
        // Handle window resize
        window.addEventListener('resize', function() {
            // Reinitialize visualizations on resize
            setTimeout(() => {
                loadConceptualSpace();
                loadKnowledgeGraph();
            }, 100);
        });
    </script>
</body>
</html>
        """
    
    async def start(self, open_browser: bool = True):
        """
        Start the dashboard server.
        
        Args:
            open_browser: Whether to automatically open browser
        """
        if open_browser:
            # Open browser after a short delay
            def open_browser_delayed():
                import time
                time.sleep(1.5)  # Wait for server to start
                webbrowser.open(f"http://{self.host}:{self.port}")
            
            import threading
            threading.Thread(target=open_browser_delayed, daemon=True).start()
        
        # Start the server
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    async def stop(self):
        """Stop the dashboard server."""
        # Close all WebSocket connections
        for connection in self.active_connections:
            try:
                await connection.close()
            except:
                pass
        
        self.active_connections.clear()


# Utility function for easy dashboard launch
async def launch_dashboard(
    kse_memory: KSEMemory,
    host: str = "localhost",
    port: int = 8080,
    open_browser: bool = True
):
    """
    Launch KSE Memory dashboard.
    
    Args:
        kse_memory: KSE Memory instance
        host: Host to bind to
        port: Port to run on
        open_browser: Whether to open browser
    """
    dashboard = KSEDashboard(kse_memory, host, port)
    await dashboard.start(open_browser)