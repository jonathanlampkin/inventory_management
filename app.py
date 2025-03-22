# Create this file in your project root
import importlib

# Import module with numeric prefix using importlib
interactive_dashboard = importlib.import_module("scripts.06_interactive_dashboard")
app = interactive_dashboard.app

server = app.server

if __name__ == '__main__':
    app.run_server(debug=False) 