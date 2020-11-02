from flask import Flask, request, jsonify
import numpy as np
from detail_model import test, preprocess, get_prediction

app = Flask(__name__)
@app.route('/')
def home():
    return "This is the emotion classification"


@app.route('/predict', methods=['GET', 'POST'])
def predict():

    data = test()
    if data:
        try:
            # 1 load message
            content = request.get_json()
            msg = content['sentance']
            # print(msg)

            # 2 proprocess the data
            prep = preprocess(msg)

            # 3 prediction
            prediction = get_prediction(prep)
            # print("here:", prediction)

            # 4 return json data
            return jsonify({'result': prediction})

        except:
            return jsonify({'result': traceback.format_exc()})

if __name__ == '__main__':
    app.run(debug=True)
