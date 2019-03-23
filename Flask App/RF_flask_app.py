import flask
from flask import request
from flask import Flask
import numpy as np
import pandas as pd
import pickle

# Loading the Model

with open('/Users/zoemarkovits/Documents/Grad School/Spring 2019/Data Mining/Project Two/pickle_jar/X.pkl', 'rb') as f:
    X = pickle.load(f)
with open('/Users/zoemarkovits/Documents/Grad School/Spring 2019/Data Mining/Project Two/pickle_jar/y.pkl', 'rb') as f:
    y = pickle.load(f)

with open('/Users/zoemarkovits/Documents/Grad School/Spring 2019/Data Mining/Project Two/pickle_jar/RF.pkl', 'rb') as f:
    RF = pickle.load(f)

predictor = RF.fit(X,y)
print(RF.score(X,y))
print(RF.score(X,y))


# Running Predictor and Webpage

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    y_pred = predictor.predict(X)

    results = {"Predicted": list(y_pred)}

    return flask.jsonify(results)

    

# Run web app server

app.run(host='0.0.0.0')
app.run(debug=True)
