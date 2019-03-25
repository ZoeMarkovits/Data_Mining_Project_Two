import flask
from flask import request
from flask import render_template
import pickle
import pandas as pd

app = flask.Flask(__name__)


# Loading the Data and Model
with open('/Users/zoemarkovits/Documents/Grad School/Spring 2019/Data Mining/Project Two/pickle_jar/X.pkl', 'rb') as f:
    X = pickle.load(f)
with open('/Users/zoemarkovits/Documents/Grad School/Spring 2019/Data Mining/Project Two/pickle_jar/y.pkl', 'rb') as f:
    y = pickle.load(f)
with open('/Users/zoemarkovits/Documents/Grad School/Spring 2019/Data Mining/Project Two/pickle_jar/RF.pkl', 'rb') as f:
    RF = pickle.load(f)

RF_fit = RF.fit(X,y)
#print(RF.score(X,y))

# Defining predict and result function
@app.route('/')
def html_page():
    with open("RF_webpage.html", 'r') as html_file:
        return html_file.read()


@app.route('/predictor', methods=['POST'])
def predictor():
    if request.method == 'POST':
        data = flask.request.json
        df = pd.DataFrame(data)
        prediction = RF_fit.predict(df)
        results = {"Predicted": list(prediction)}
        return flask.jsonify(results)
    return render_template('RF_result.html',prediction=prediction)

"""
@app.route('/predictor', methods=['POST'])
def predictor(input):
    to_predict = np.array(input).reshape(1,6)
    result = RF_fit.predict(to_predict)
    return result[0]

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        input = request.form.to_dict()
        input = list(input.values())
        input = list(map(int, input))
        result = predictor(input)
        if int(result)==1:
            prediction='Academic Performance High'
        else:
            prediction='Academic Performance Low'
        return render_template("RF_result.html",prediction=prediction)
"""


# Run web app server
if __name__ =='__main__':
    app.run(debug=True)
