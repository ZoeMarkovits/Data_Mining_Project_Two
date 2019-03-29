# Defining predict and result function
"""
@app.route('/')
def html_page():
    with open("RF_webpage.html", 'r') as html_file:
        return html_file.read()


@app.route('/predictor', methods=['POST'])
def predictor():
    data = flask.request.json
    df = pd.DataFrame(data)
    prediction = RF_fit.predict(df)
    results = {"Predicted": list(prediction)}
    return flask.jsonify(results)
    #return render_template('RF_result.html',prediction=prediction)
"""
