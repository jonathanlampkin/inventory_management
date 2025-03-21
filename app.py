# Create this file in your project root
from scripts.06_interactive_dashboard import app

server = app.server

if __name__ == '__main__':
    app.run_server(debug=False) 