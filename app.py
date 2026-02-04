import pickle
from flask import Flask, request, jsonify, app, render_template, url_for
import numpy as np
import pandas as pd

app = Flask(__name__)
#Load the model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    data_df = pd.DataFrame(data)
    print(data_df)
    prediction = regmodel.predict(data_df)
    print('Prediction is:', prediction)
    return jsonify(prediction.tolist())