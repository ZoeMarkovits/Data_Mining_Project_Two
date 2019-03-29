import flask
from flask import request
from flask import render_template
import pickle
import pandas as pd
import numpy as np

app = flask.Flask(__name__)


# Loading the Data and Model
with open('/Users/zoemarkovits/Documents/Grad School/Spring 2019/Data Mining/Project Two/pickle_jar/X.pkl', 'rb') as f:
    X = pickle.load(f)
with open('/Users/zoemarkovits/Documents/Grad School/Spring 2019/Data Mining/Project Two/pickle_jar/y.pkl', 'rb') as f:
    y = pickle.load(f)
with open('/Users/zoemarkovits/Documents/Grad School/Spring 2019/Data Mining/Project Two/pickle_jar/RF.pkl', 'rb') as f:
    RF = pickle.load(f)

# X columns: Raised_Hand, Visited_Resources, Viewed_Announcements, Discussion_Groups, Under_Seven (Absences)

RF_fit = RF.fit(X,y)
RF_fit.predict(X)
RF_fit.score(X,y)

tester_example = X[:1]
tester_example
RF.predict(tester_example)

@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,5)
    loaded_model = RF
    result = loaded_model.predict(to_predict)
    return result[0]

ValuePredictor(tester_example)

@app.route('/result',methods = ['POST'])
def result():
    #if request.method == 'POST':
    to_predict_list = request.form.to_dict()
    to_predict_list=list(to_predict_list.values())
    to_predict_list = list(map(int, to_predict_list))
    result = ValuePredictor(to_predict_list)
    if int(result)==1:
        prediction='Prediction: Performance is High'
    else:
        prediction='Prediction: Performance is Low'
    return render_template("result.html",prediction=prediction)

# Run web app server
if __name__ == '__main__':
    app.run(debug=True)
