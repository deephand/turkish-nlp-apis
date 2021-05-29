from flask import Flask
from .nlp import routes as nlp_routes


def create_app():
    app = Flask(__name__)

    app.register_blueprint(nlp_routes.app)
    
    return app
