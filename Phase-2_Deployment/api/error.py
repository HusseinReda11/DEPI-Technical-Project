"""
Error handlers for the API endpoints.
"""
from flask import Blueprint, jsonify
from werkzeug.exceptions import HTTPException
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Create blueprint
errors_bp = Blueprint('errors', __name__)

@errors_bp.app_errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    logger.warning(f"404 error: {error}")
    return jsonify({'error': 'Resource not found'}), 404

@errors_bp.app_errorhandler(400)
def bad_request(error):
    """Handle 400 errors."""
    logger.warning(f"400 error: {error}")
    return jsonify({'error': 'Bad request'}), 400

@errors_bp.app_errorhandler(500)
def internal_server_error(error):
    """Handle 500 errors."""
    logger.error(f"500 error: {error}", exc_info=True)
    return jsonify({'error': 'Internal server error'}), 500

@errors_bp.app_errorhandler(Exception)
def handle_exception(error):
    """Handle all other exceptions."""
    logger.error(f"Unhandled exception: {error}", exc_info=True)
    
    # If it's an HTTPException, return its response
    if isinstance(error, HTTPException):
        return jsonify({'error': error.description}), error.code
    
    # Otherwise return a 500 error
    return jsonify({'error': 'Internal server error'}), 500