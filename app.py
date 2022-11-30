import pickle
import sklearn
from flask import Flask, request, jsonify, render_template

import pandas as pd
import numpy as np

app: Flask=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/', methods=['GET'])
def homepage():
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    new_data = [list(data.values())]
    output = model.predict(new_data)[0]
    return jsonify(output)

@app.route('/predict_api2', methods = ['POST'])
def predict_api2():
    data = [float(x) for x in request.form.values()]
    final_features = [np.array(data)]
    output = model.predict(final_features)[0]
    print(output)
    return render_template('home.html', prediction_text='Airfoil pressure is {}'.format(output))

if __name__=='__main__':
    app.run(debug=True)