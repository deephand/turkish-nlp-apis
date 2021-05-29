from flask import Blueprint
from .stemmer import routes as stemmer_routes

NAME = 'nlp'

app = Blueprint(NAME, __name__, url_prefix=f'/{NAME}')
app.register_blueprint(stemmer_routes.app)