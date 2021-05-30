# Turkish NLP APIs
This is a repo where I plan to put my experimental models behind a REST API to be deployed in Heroku. It is a Python Flask server running behind gunicorn. 

### Current APIs:
1. `/nlp/stemmer/q=**words**`: returns the stems of **words** using a Seq2Seq Neural Model
   1. Params:
      - **words** (str): contains 1+ words. Special characters like quote, question mark will be discarded
   1. Returns:
      - stemmed **words** (str). Not wrapped in any JSON or similar for now.
   1. Model details:
      - Characters are one hot encoded.
      - 2 layers of Bidirectional GRUs with 32 hidden units
      - 2 layers of GRUs with 64 hidden units, each using the corresponding state of the encoder
      - 78k params
      - Model is trained in personal Colab (https://colab.research.google.com/drive/1SjuaADaHocVkNfgeY0iYsKj_c1sAm7KK)
   1. Training data details:
      - I once found a Turkish corpus from Zargan containing words, their roots, freqs, etc. Now it's not online. I used 50k word pairs for training & testing.
   1. Example Usages (already deployed):
      - https://cryptic-mountain-83365.herokuapp.com/nlp/stemmer/q=urfanın%20etrafı%20dumanlı%20dağlar
      - https://cryptic-mountain-83365.herokuapp.com/nlp/stemmer/q=kayserilileştiremediklerimizdenmişsiniz
 
### Setup:

1. `curl https://cli-assets.heroku.com/install.sh | sh` # https://devcenter.heroku.com/articles/heroku-cli#standalone-installation
1. `heroku login --interactive`
1. `python -m venv venv`
1. `. venv/bin/activate`
1. `pip install -r requirements.txt`
1. `heroku create`
1. `git push heroku main`

Local testing:

1. `heroku local`


