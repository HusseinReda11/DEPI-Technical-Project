"""
Main Flask application for the sentiment analysis service.
"""
import logging
from flask import Flask, render_template
from flask_cors import CORS
from pathlib import Path
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def create_app():
    """
    Create and configure the Flask application.
    
    Returns:
        Configured Flask app
    """
    # Create Flask app
    app = Flask(__name__)
    
    # Enable CORS
    CORS(app)
    
    # Load configuration
    app.config.from_pyfile('config.py')
    
    # Register blueprints
    from api import api_bp, errors_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(errors_bp)
    
    # Root route
    @app.route('/')
    def index():
        """Render the main application page."""
        return render_template('index.html')
    
    # Log startup information
    logger.info(f"Application created with debug={app.config['DEBUG']}")
    
    return app

# Create the application
app = create_app()

if __name__ == '__main__':
    logger.info(f"Starting server on {app.config['HOST']}:{app.config['PORT']}")
    app.run(
        host=app.config['HOST'],
        port=app.config['PORT'],
        debug=app.config['DEBUG']
    )