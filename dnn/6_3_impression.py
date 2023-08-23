import pandas as pd
import tensorflow  as tf
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Model
from tensorflow.keras.layers import  Input, Embedding, Dense, Dropout, Conv1D, GlobalMaxPool1D, concatenate

train_file = './data/ChatbotData.csv'
data = pd.read_csv(train_file, delimiter=',')
features = data['Q'].tolist()
labels = data['label'].tolist()

corpus = [preprocessing.text.text_to_word_sequence(text) for text in features] #문장을 단어시퀀스로 만듦, 리스트에 저장
tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus) #단어들을 시퀀스 번호로 변환(희소표현)
word_index = tokenizer.word_index

MAX_SEQ_LEN = 15
padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding = 'post') #padding
ds = tf.data.Dataset.from_tensor_slices((padded_seqs, labels))
ds = ds.shuffle(len(features))

train_size = int(len(padded_seqs) * 0.7)
val_size = int(len(padded_seqs) * 0.2)
test_size = int(len(padded_seqs) * 0.1)

train_ds = ds.take(train_size).batch(20)
val_ds = ds.skip(train_size).take(val_size).batch(20)
test_ds = ds.skip(train_size + val_size).take(test_size).batch(20)

dropout_prob = 0.5 #dropout 설정, 출력 유닛 일부 비활성화-> 노이즈 추가, 과적합 방지
EMB_SIZE = 128 #밀집표현 차원 크기
EPOCH = 5 # 반복 횟수
VOCAB_SIZE = len(word_index) + 1

input_layer = Input(shape=(MAX_SEQ_LEN,))
embedding_layer = Embedding(VOCAB_SIZE, EMB_SIZE, input_length = MAX_SEQ_LEN)(input_layer)
dropout_emb = Dropout(rate=dropout_prob)(embedding_layer) 

conv1 = Conv1D(filters = 128, kernel_size = 3, padding = 'valid', activation = tf.nn.relu)(dropout_emb) #필터 수 128, 3-gram
pool1 = GlobalMaxPool1D()(conv1) #최대 풀링

conv2 = Conv1D(filters = 128, kernel_size = 4, padding = 'valid', activation = tf.nn.relu)(dropout_emb) # 4-gram
pool2 = GlobalMaxPool1D()(conv2)

conv3 = Conv1D(filters = 128, kernel_size = 5, padding = 'valid', activation = tf.nn.relu)(dropout_emb) # 5-gram
pool3 = GlobalMaxPool1D()(conv3)

concat = concatenate([pool1, pool2, pool3])

hidden = Dense(128, activation = tf.nn.relu)(concat)
dropout_hidden = Dropout(rate = dropout_prob)(hidden)
logits = Dense(3, name='logits')(dropout_hidden)
predictions = Dense(3, activation = tf.nn.softmax)(logits)

model = Model(inputs = input_layer, outputs = predictions)
model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_ds, validation_data = val_ds, epochs = EPOCH, verbose = 1)

loss, accuracy = model.evaluate(test_ds, verbose = 1)
print('Accuracy: {}%'.format(accuracy * 100))
print('loss: {}%'.format(loss * 100))

model.save('./models/cnn_model.keras')
