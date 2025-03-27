#!/usr/bin/env python3
"""
A simple proxy server to connect the DARF frontend with the API server
"""
import http.server
import socketserver
import urllib.parse
import requests
import json
from urllib.parse import urljoin
import sys

# Configuration
FRONTEND_PORT = 3004  # Port for the frontend
API_URL = "http://localhost:5000"  # URL of the running API server

class ProxyHandler(http.server.SimpleHTTPRequestHandler):
    """Handler that serves static files and proxies API requests"""
    
    def do_GET(self):
        """Handle GET requests"""
        # If this is an API request, proxy it
        if self.path.startswith('/api/'):
            self.proxy_request('GET')
        else:
            # Otherwise, serve static files from the frontend/build directory
            super().do_GET()
    
    def do_POST(self):
        """Handle POST requests"""
        # All POST requests are proxied to the API server
        self.proxy_request('POST')
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests (for CORS)"""
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()
    
    def send_cors_headers(self):
        """Add CORS headers to response"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
    
    def end_headers(self):
        """Add CORS headers to all responses"""
        self.send_cors_headers()
        super().end_headers()
    
    def proxy_request(self, method):
        """Proxy a request to the API server"""
        try:
            # Get the API endpoint from the path
            api_path = self.path
            target_url = urljoin(API_URL, api_path)
            
            print(f"Proxying {method} request to {target_url}")
            
            # Get request headers and body
            headers = {key: val for key, val in self.headers.items()}
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length) if content_length > 0 else None
            
            # Make the request to the API server
            response = requests.request(
                method=method,
                url=target_url,
                headers=headers,
                data=body,
                timeout=10
            )
            
            # Send the response back to the client
            self.send_response(response.status_code)
            
            # Add response headers
            for key, val in response.headers.items():
                if key.lower() not in ('transfer-encoding', 'content-encoding'):
                    self.send_header(key, val)
            
            self.end_headers()
            
            # Send response body
            self.wfile.write(response.content)
            
        except Exception as e:
            print(f"Error proxying request: {e}")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

def main():
    """Main function to start the proxy server"""
    # Change to frontend/build directory to serve static files
    import os
    os.chdir("darf_frontend/build")
    
    # Start the server
    handler = ProxyHandler
    httpd = socketserver.ThreadingTCPServer(("", FRONTEND_PORT), handler)
    
    print(f"Starting proxy server on port {FRONTEND_PORT}")
    print(f"Serving frontend files from: {os.getcwd()}")
    print(f"Proxying API requests to: {API_URL}")
    print("\nAccess the DARF Framework at: http://localhost:3004\n")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.shutdown()

if __name__ == "__main__":
    main()
