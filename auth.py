import os
import json
import uuid
import logging
import functools
from flask import request, jsonify, g
from werkzeug.security import generate_password_hash, check_password_hash

logger = logging.getLogger(__name__)

# Store for API tokens
api_tokens = {}

# Store for users
users = {}

def _load_api_tokens():
    """Load API tokens from file."""
    global api_tokens
    try:
        token_file = os.environ.get("API_TOKENS_FILE", "api_tokens.json")
        if os.path.exists(token_file):
            with open(token_file, "r") as f:
                api_tokens = json.load(f)
        else:
            logger.warning(f"API tokens file not found: {token_file}")
            # Create default development token
            token = str(uuid.uuid4()).replace("-", "")
            api_tokens = {"dev": {"token": token, "role": "admin"}}
            logger.info(f"Created default development token: {token}")
            _save_api_tokens()
    except Exception as e:
        logger.error(f"Error loading API tokens: {e}")

def _save_api_tokens():
    """Save API tokens to file."""
    try:
        token_file = os.environ.get("API_TOKENS_FILE", "api_tokens.json")
        with open(token_file, "w") as f:
            json.dump(api_tokens, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving API tokens: {e}")

def _load_users():
    """Load users from file."""
    global users
    try:
        users_file = "users.json"
        if os.path.exists(users_file):
            with open(users_file, "r") as f:
                users = json.load(f)
        else:
            logger.warning(f"Users file not found: {users_file}")
            # Create default admin user
            users = {
                "admin": {
                    "password_hash": generate_password_hash("admin"),
                    "role": "admin"
                }
            }
            logger.info("Created default admin user (username: admin password: admin)")
            _save_users()
    except Exception as e:
        logger.error(f"Error loading users: {e}")

def _save_users():
    """Save users to file."""
    try:
        users_file = "users.json"
        with open(users_file, "w") as f:
            json.dump(users, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving users: {e}")

# Initialize tokens and users
_load_api_tokens()
_load_users()

def darf_auth():
    """Authentication middleware for Flask."""
    # Check for API token
    token = request.headers.get("Authorization")
    if token and token.startswith("Bearer "):
        token = token[7:]  # Remove "Bearer " prefix
        for user_id, user_data in api_tokens.items():
            if user_data.get("token") == token:
                g.user = user_id
                g.role = user_data.get("role", "user")
                return
    
    # If no valid token, set user to None
    g.user = None
    g.role = None

def require_auth(f):
    """Decorator to require authentication."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not g.user:
            return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)
    return decorated

def require_role(role):
    """Decorator to require a specific role."""
    def decorator(f):
        @functools.wraps(f)
        def decorated(*args, **kwargs):
            if not g.user:
                return jsonify({"error": "Authentication required"}), 401
            if g.role != role and g.role != "admin":
                return jsonify({"error": "Insufficient permissions"}), 403
            return f(*args, **kwargs)
        return decorated
    return decorator

