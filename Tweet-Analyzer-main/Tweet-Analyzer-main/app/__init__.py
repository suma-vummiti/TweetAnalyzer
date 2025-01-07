# app/_init_.py
from flask import Flask

def create_app():
    app = Flask(__name__)

    # Import routes after the app is created
    from app.routes import index, results  # Import the views/routes

    # Register routes
    app.add_url_rule('/', 'index', index)
    app.add_url_rule('/results', 'results', results, methods=['POST'])

    return app