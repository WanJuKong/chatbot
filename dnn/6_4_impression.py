import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import preprocessing

train_file = './data/ChatbotData.csv'
data = pd.read_csv(train_file, delimiter=',')
features = data['Q'].tolist()
labels = data['label'].tolist()

corpus = [preprocessing.text.text_to_word_sequence(text) for text in features]
tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)

MAX_SEQ_LEN = 15
padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post')

ds = tf.data.Dataset.from_tensor_slices((padded_seqs, labels))
ds = ds.shuffle(len(features))
test_ds = ds.take(2000).batch(20)

model = load_model('./models/cnn_model.keras')
model.summary()
model.evaluate(test_ds, verbose = 2)

DATA_INDEX = int(input('index(0~{}) : '.format(len(corpus)-1)))

print('단어 시퀀스: ', corpus[DATA_INDEX])
print('단어 인덱스 시퀀스: ', padded_seqs[DATA_INDEX])
print('문장 분류: ', labels[DATA_INDEX])

picks = [DATA_INDEX]
predict = model.predict(padded_seqs[picks])
predict_class = tf.math.argmax(predict, axis=1)
print('감정 예측 접수: ', predict)
print('에측 클래스: ', predict_class.numpy())

