from flask import Flask, request, jsonify
from detail_model import preprocess, get_prediction

app = Flask(__name__)
@app.route('/')
def home():
    return "This is the emotion classification"


@app.route('/predict', methods=['GET', 'POST'])
def predict():

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
        return jsonify({'result': "Sorry try it again"})


if __name__ == '__main__':
    app.run(debug=True)
