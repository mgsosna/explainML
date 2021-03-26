from flask import Flask, render_template, request, jsonify
from static import Model

import json
import pandas as pd


DATA_PATH = "static/data/exam_data.csv"
FEATURES = ["sleep", "study"]
TARGET = "exam_score"

TRAIN_DATA = pd.read_csv(DATA_PATH)

mod = Model()
mod.initialize(TRAIN_DATA, FEATURES, TARGET)


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/<string:modtype>', methods=['POST'])
def predict(modtype):

    input_dict = request.get_json()

    if not isinstance(input_dict[FEATURES[0]], list):
        input_df = pd.DataFrame(input_dict, index=[0])
    else:
        input_df = pd.DataFrame(input_dict)

    if modtype == 'linear':
        pred = mod.lr.predict(input_df)
        return jsonify(mod.lr.predict(input_df).tolist())
    elif modtype == 'random_forest':
        return jsonify(mod.rf.predict(input_df).tolist())

    return jsonify(f"Error: modtype is {modtype}; must be 'linear' or 'random_forest'"), 404


if __name__ == '__main__':
    app.run(debug=True, port=5000)
