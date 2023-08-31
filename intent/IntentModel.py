# 8_8

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import preprocessing

class IntentModel:
    def __init__(self, model, preprocess):
        self.labels = {0 : '인사', 1 : '주문', 2 : '기타'}
        self.model = load_model(model)
        self.p = preprocess
        
    def predict_class(self, query):
        pos = self.p.pos(query)

        MAX_SEQ_LEN = 15

        keywords = self.p.get_keywords(pos, without_tag = True)
        sequences = [self.p.get_wordidx_sequence(keywords)]

        padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post')

        predict = self.model.predict(padded_seqs)
        predict_class = tf.math.argmax(predict, axis=1)
        return predict_class.numpy()[0]
    
    def class_probabilities(self, query):
        pos = self.p.pos(query)
        
        MAX_SEQ_LEN=15

        keywords = self.p.get_keywords(pos, without_tag = True)
        sequences = [self.p.get_wordidx_sequence(keywords)]

        padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post')

        predict = self.model.predict(padded_seqs)
        class_probabilities = tf.math.softmax(predict[0])
        return class_probabilities.numpy().tolist()
