import logging
from flask import jsonify, Flask

logger = logging.getLogger(__name__)

class DARFAPIError(Exception):
    """Base class for all DARF API errors."""
    status_code = 500
    
    def __init__(self, message, status_code=None, payload=None):
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload
    
    def to_dict(self):
        rv = dict(self.payload or {})
        rv["error"] = self.message
        rv["status_code"] = self.status_code
        return rv

class ValidationError(DARFAPIError):
    """Raised when request validation fails."""
    status_code = 400

class ResourceNotFoundError(DARFAPIError):
    """Raised when a requested resource is not found."""
    status_code = 404

class LLMError(DARFAPIError):
    """Raised when there is an error with the LLM."""
    status_code = 500

def register_error_handlers(app):
    """Register error handlers with the Flask app."""
    
    @app.errorhandler(DARFAPIError)
    def handle_darf_api_error(error):
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response
    
    @app.errorhandler(404)
    def handle_not_found(error):
        return jsonify({"error": "Resource not found", "status_code": 404}), 404
    
    @app.errorhandler(500)
    def handle_server_error(error):
        logger.exception("Unhandled exception")
        return jsonify({"error": "Internal server error", "status_code": 500}), 500

def validate_request(data, required_fields):
    """Validate that required fields are present in the request data."""
    if not data:
        raise ValidationError("No data provided")
    
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValidationError(f"Missing required fields: {missing_fields}")

