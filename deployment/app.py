# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 18:17:49 2021

@author: asaga
"""


#loading libraries
import joblib
from flask import Flask, request
from get_prediction import get_recommendations

# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)
# render_template

@app.route('/')
def home():

    """Serve homepage template."""
    return flask.render_template('index.html')
    #return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    to_predict_list = request.form.to_dict()
    #print(to_predict_list)
    predictions, time = get_recommendations(to_predict_list)
    print(predictions, time)
    if 'recommend' not in predictions.keys():
        #return flask.redirect('new_user_recommendation.html',predictions = predictions)
        return flask.render_template("new_user_recommendation.html",predictions = predictions)

    return flask.render_template("predict.html",predictions = predictions)
    #return jsonify({'products': recommended_products, 'Time': difference, 'predict_list':to_predict_list, 'top5':top5_products})


if __name__ == '__main__':
    #app.debug = True
    app.run(host='0.0.0.0', port=8080)
