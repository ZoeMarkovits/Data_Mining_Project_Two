import flask
from flask import Flask
import pickle

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

X


# Defining predict function

@app.route('/')
def html_page():
    with open("RF_flask_app.html", 'r') as html_file:
        return html_file.read()

@app.route('/predict', methods = ['POST'])
def predict():
    data = flask.request.json
    x = np.array(data["example"]).reshape(-1,1)

    y_pred = RF_fit.predict(x)
    results = {"Predicted": list(y_pred)}
    return flask.jsonify(results)


# Run web app server
if __name__ =='__main__':
    app.run(debug=True)
