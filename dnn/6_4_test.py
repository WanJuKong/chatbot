import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import preprocessing

train_file = './data/ChatbotData.csv'
data = pd.read_csv(train_file, delimiter=',')
features = data['Q'].tolist()
labels = data['label'].tolist()

input_text = input('Input text: ')
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

print('Input words: ', input_words)
print('Padded input sequence: ', padded_input_seq)

predict = model.predict(padded_input_seq)
predict_class = tf.math.argmax(predict, axis=1)
print('Emotion prediction probabilities: ', predict)
print('Predicted class: ', predict_class.numpy())
