from flask import Flask,request,render_template
import pickle
import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBClassifier
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split as tts
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, confusion_matrix

app = Flask(__name__)
model = pickle.load(open('clf_healthcare.pkl', 'rb'))


@app.route('/')
def hello_world():
    return render_template('health.html')


@app.route('/predict', methods=['POST','GET'])
def predict():
    
    int_features = [float(x) for x in request.form.values()]
    final = [np.array(int_features)]
    
    prediction = model.predict(final)
    return render_template('health.html', pred="Your Output is : {0}".format(prediction))


if __name__ == 'main':
    app.run()

