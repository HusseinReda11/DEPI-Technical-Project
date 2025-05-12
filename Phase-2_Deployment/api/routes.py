"""
API route definitions for the sentiment analysis application.
"""
from flask import Blueprint, request, jsonify, current_app
from typing import Dict, Any, List
import logging

from models import get_model

# Set up logging
logger = logging.getLogger(__name__)

# Create blueprint
api_bp = Blueprint('api', __name__)

@api_bp.route('/predict', methods=['POST'])
def predict() -> Dict[str, Any]:
    """
    Endpoint for sentiment prediction.
    
    Expects a JSON payload with a 'tweets' field containing a list of objects
    with 'text' fields.
    
    Returns:
        JSON response with prediction results
    """
    try:
        data = request.get_json()
        
        if not data:
            logger.warning("No JSON data in request")
            return jsonify({'error': 'No JSON data provided'}), 400
        
        tweets = data.get('tweets', None)
        
        if tweets is None:
            logger.warning("Missing 'tweets' field in request")
            return jsonify({'error': 'Missing "tweets" field'}), 400
        
        if not isinstance(tweets, list):
            logger.warning("'tweets' field is not a list")
            return jsonify({'error': '"tweets" field must be a list'}), 400
        
        # Log the request (but not in production with sensitive data)
        if current_app.debug:
            logger.debug(f"Prediction request: {tweets}")
        
        # Get the model and make predictions
        model = get_model()
        results = model.predict(tweets)
        
        # Log the response (but not in production with sensitive data)
        if current_app.debug:
            logger.debug(f"Prediction response: {results}")
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Error in /predict endpoint: {e}", exc_info=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@api_bp.route('/healthz', methods=['GET'])
def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    
    Returns:
        JSON response with status
    """
    return jsonify({'status': 'ok'})