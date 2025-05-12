"""
WSGI entry point for production deployment.
"""
from server import app

if __name__ == '__main__':
    app.run()