import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import preprocessing

train_file = './data/ChatbotData.csv'
data = pd.read_csv(train_file, delimiter=',')
features = data['Q'].tolist()
labels = data['label'].tolist()

input_text = input('input text: ')
features.append(input_text)
corpus = [preprocessing.text.text_to_word_sequence(text) for text in features]
tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)

MAX_SEQ_LEN = 15

input_words = preprocessing.text.text_to_word_sequence(input_text)
input_seq = tokenizer.texts_to_sequences([input_words])
padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post')
padded_input_seq = preprocessing.sequence.pad_sequences(input_seq, maxlen=MAX_SEQ_LEN, padding='post')

model = load_model('./models/cnn_model.keras')
model.summary()


print('단어 시퀀스: ', input_words)
print('단어 인덱스 시퀀스: ', padded_input_seq)

predict = model.predict(padded_input_seq)
predict_class = tf.math.argmax(predict, axis=1)
print('감정 예측 접수: ', predict)
print('에측 클래스: ', predict_class.numpy())

