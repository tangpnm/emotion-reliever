from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

#load model
path_model = './model/LSTM_w2v.h5'

# Loading the model and assign it to the predictor
loaded_model = load_model(path_model)

#Initialize the value
tokenizer = Tokenizer()
max_seq_len = 500
class_names = ['joy', 'fear', 'anger', 'sadness', 'neutral']

def preprocess(message):

    seq = tokenizer.texts_to_sequences(message)
    padded = pad_sequences(seq, maxlen=max_seq_len)

    return padded


def get_prediction(padded):

    predict = loaded_model.predict(padded)
    answer = class_names[np.argmax(predict)]

    return answer
