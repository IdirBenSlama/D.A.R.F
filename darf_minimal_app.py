#!/usr/bin/env python
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DARF Framework Status</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
                text-align: center;
            }
            h1 {
                color: #333;
                margin-bottom: 20px;
            }
            .status-box {
                background-color: #f8f9fa;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 20px;
                margin-top: 20px;
                text-align: left;
            }
            .status-indicator {
                background-color: #e6ffed;
                border-left: 4px solid #2cbe4e;
                padding: 10px 15px;
                margin-bottom: 15px;
            }
            .components {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
                margin-top: 20px;
            }
            .component {
                background-color: #f0f7ff;
                border: 1px solid #cce5ff;
                border-radius: 4px;
                padding: 10px;
            }
        </style>
    </head>
    <body>
        <h1>DARF Framework</h1>
        
        <div class="status-box">
            <div class="status-indicator">
                <strong>Status:</strong> Running
            </div>
            
            <h3>System Information</h3>
            <p>The Decentralized Autonomous Reaction Framework is running in standard mode.</p>
            
            <h3>Active Components</h3>
            <div class="components">
                <div class="component">
                    <strong>Knowledge Graph Engine</strong><br>
                    Status: Active with demo data
                </div>
                <div class="component">
                    <strong>Event Bus</strong><br>
                    Status: Running
                </div>
                <div class="component">
                    <strong>LLM Interface</strong><br>
                    Status: Initialized
                </div>
                <div class="component">
                    <strong>Web UI</strong><br>
                    Status: Running on port 5000
                </div>
            </div>
        </div>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)
