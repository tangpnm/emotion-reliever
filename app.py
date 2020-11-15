from flask import Flask, request, jsonify
import nltk
nltk.download('punkt')

import pandas as pd
from nltk.tokenize import word_tokenize
import re
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

app = Flask(__name__)
@app.route('/')
def home():
    return "This is the emotion classification"


@app.route('/predict', methods=['GET', 'POST'])
def predict():

    # load model
    path_model = './model/model_ver2.h5'

    # Loading the model and assign it to the predictor
    loaded_model = load_model(path_model)

    #Initial value
    max_seq_len = 500
    class_names = ['joy', 'fear', 'anger', 'sadness', 'neutral']

    #Initial word tokenize
    data_train = pd.read_csv(
        './model/data_train.csv', encoding='utf-8')
    data_test = pd.read_csv(
        './model/data_test.csv', encoding='utf-8')
    X_train = data_train.Text
    X_test = data_test.Text
    data = data_train.append(data_test, ignore_index=True)

    def clean_text(data):
        # remove hashtags and @usernames
        data = re.sub(r"(#[\d\w\.]+)", '', data)
        data = re.sub(r"(@[\d\w\.]+)", '', data)

        # tekenization using nltk
        data = word_tokenize(data)

        return data

    #preprocess
    texts = [' '.join(clean_text(text)) for text in data.Text]
    texts_train = [' '.join(clean_text(text)) for text in X_train]
    texts_test = [' '.join(clean_text(text)) for text in X_test]
    tokenizer = Tokenizer()
    
    tokenizer.fit_on_texts(texts)
    sequence_train = tokenizer.texts_to_sequences(texts_train)
    sequence_test = tokenizer.texts_to_sequences(texts_test)
    
    try:
   
        #1 load data
        content = request.get_json()
        msg = [content['sentance']]

        #2 data preprocessing
        seq = tokenizer.texts_to_sequences(msg)
        padded = pad_sequences(seq, maxlen=max_seq_len)

        #3 data prediction
        pred = loaded_model.predict(padded)
        answer = class_names[np.argmax(pred)]

        return jsonify({'result': answer})
    except:
        return jsonify({'result': "sorry try it again"})

if __name__ == '__main__':
    app.run(debug=True)
