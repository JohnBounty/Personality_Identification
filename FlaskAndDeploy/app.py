# =============================Dependencies=============================
# base
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os
import time

# personality insight
from ibm_watson import PersonalityInsightsV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import joblib

# # watson machine learning
# from watson_machine_learning_client import WatsonMachineLearningAPIClient

# custom
import app_func.aux_func
from app_func.aux_func import get_insight_local

# flask
from flask import Flask, send_file
from flask import request, render_template, jsonify
IMG_FOLDER = "static/images/"

app = Flask(__name__)
# app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = IMG_FOLDER
matplotlib.use('agg')

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache"
    return r

@app.route('/')
def home():
    if os.path.exists(IMG_FOLDER+'radar_plot'):
        os.remove(IMG_FOLDER+'radar_plot')
    return render_template('home.html')

@app.route('/joined', methods=['GET', 'POST'])
def my_form_post():
    text_input = request.form['text_input']
    word = request.args.get('text_input')
    prediction = get_insight_local(text_input)
    response = 'prediction_{:d}.html'.format(prediction)
    print(response)
    return render_template(response)


if __name__ == '__main__':

    app.run(host='0.0.0.0', debug=False)


# @app.route('/what', methods=['GET'])
# def regr_plot():
#     image = regression_plot()
#
#     return send_file(image,
#     attachment_filename='regplot.png',
#     mimetype='image/png')

    # to_send = "In my opinion, this dude belongs to class {:d}.".format(prediction[0])
    # result = {
    #     "output": to_send
    # }
    # result = {str(key): value for key, value in result.items()}
    # return jsonify(result=result)


        # r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0
