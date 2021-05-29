from flask import session, Blueprint
from .seq2seq.model import Model


app = Blueprint('stemmer', __name__, url_prefix='/stemmer')
model = Model()

@app.route('/')
def hello_world():
    return '''
    <h3>Usage:</h3>
    <p> q=[words] </p>
    <p> words: keywords to be stemmed, separated with html decoded spaces (%20) </p>
    '''

@app.route('/q=<words>')
@app.route('/q=<words>&model=<model_name>')
def stem_word(words, model_name='seq2seq'):
    if model_name is None:
        return f'<p>Not implemented yet!</p>'

    global model
    stem = model.predict(words)

    return f'<p>{stem}</p>'




        