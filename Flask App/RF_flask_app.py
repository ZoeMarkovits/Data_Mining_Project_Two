import flask
from flask import Flask
from flask import request
import pickle
import pandas as pd

app = Flask(__name__)


# Loading the Model
with open('/Users/zoemarkovits/Documents/Grad School/Spring 2019/Data Mining/Project Two/pickle_jar/X.pkl', 'rb') as f:
    X = pickle.load(f)
with open('/Users/zoemarkovits/Documents/Grad School/Spring 2019/Data Mining/Project Two/pickle_jar/y.pkl', 'rb') as f:
    y = pickle.load(f)
with open('/Users/zoemarkovits/Documents/Grad School/Spring 2019/Data Mining/Project Two/pickle_jar/RF.pkl', 'rb') as f:
    RF = pickle.load(f)

RF_fit = RF.fit(X,y)
print(RF.score(X,y))
print(RF.score(X,y))


# Defining predict function
@app.route('/')
def html_page():
    with open("RF_flask_app.html", 'r') as html_file:
        return html_file.read()

@app.route('/predict', methods = ['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data)

    prediction = RF_fit.predict(df)
    results = {"Predicted": list(prediction)}
    return flask.jsonify(results)


# Run web app server
if __name__ =='__main__':
    app.run(debug=True)
