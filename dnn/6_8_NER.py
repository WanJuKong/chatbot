import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np

def read_file(file):
    sents = []
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            if line[0] == ';' and lines[index + 1][0] == '$':
                this_sent = []
            elif line[0] == '$' and lines[index -1][0] == ';':
                continue
            elif line[0] == '\n':
                sents.append(this_sent)
            else:
                this_sent.append(tuple(line.split('\t')))
    return sents

corpus = read_file('./data/train.txt')
sentences, tags = [], []
for text in corpus:
    tagged_sentence = []
    sentence, bio_tag = [], []
    for word in text:
        tagged_sentence = []
        sentence.append(word[1])
        bio_tag.append(word[3])
    sentences.append(sentence)
    tags.append(bio_tag)
print('sample size: ', len(sentences))

sent_tokenizer = preprocessing.text.Tokenizer(oov_token='OOV')
sent_tokenizer.fit_on_texts(sentences)
tag_tokenizer = preprocessing.text.Tokenizer(lower=False)
tag_tokenizer.fit_on_texts(tags)

vocab_size = len(sent_tokenizer.word_index) + 1
tag_size = len(tag_tokenizer.word_index) + 1
print('BIO tag size: ', tag_size)
print('vocab size: ', vocab_size)

x_train = sent_tokenizer.texts_to_sequences(sentences)
y_train = tag_tokenizer.texts_to_sequences(tags)

index_to_word = sent_tokenizer.index_word
index_to_ner = tag_tokenizer.index_word
index_to_ner[0] = 'PAD'

max_len = 40
x_train = preprocessing.sequence.pad_sequences(x_train, padding='post', maxlen=max_len)
y_train = preprocessing.sequence.pad_sequences(y_train, padding='post', maxlen=max_len)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2, random_state=0)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=tag_size)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=tag_size)

print('learn sequence shape: ', x_train.shape)
print('learn tag shape: ', y_train.shape)
print('test sequence shape: ', x_test.shape)
print('test tag shape: ', y_test.shape)

######################################################

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=30, input_length=max_len, mask_zero=True))
model.add(Bidirectional(LSTM(200, return_sequences=True, dropout = 0.5, recurrent_dropout  = 0.25)))
model.add(TimeDistributed(Dense(tag_size, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.01), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=10)

print(model.evaluate(x_test, y_test)[1])

def sequence_to_tag(sequences):
    result = []
    for sequence in sequences:
        temp = []
        for pred  in sequence:
            pred_index = np.argmax(pred)
            temp.append(index_to_ner[pred_index].replace('PAD', 'O'))
        result.append(temp)
    return result

y_predicted = model.predict(x_test)
pred_tags = sequence_to_tag(y_predicted)
test_tags = sequence_to_tag(y_test)

from seqeval.metrics import f1_score, classification_report

print(classification_report(test_tags, pred_tags))
print('F1-score: {:.1%}'.format(f1_score(test_tags, pred_tags)))
word_to_index = sent_tokenizer.word_index
new_sentence = input(':').split()
new_x = []
for w in new_sentence:
    try:
        new_x.append(word_to_index.get(w, 1))
    except KeyError:
        new_x.append(word_to_index['OOV'])

print('new sequence: ', new_x)
new_padded_seqs = preprocessing.sequence.pad_sequences([new_x], padding='post', value=0, maxlen=max_len)

p = model.predict(np.array([new_padded_seqs[0]]))
p = np.argmax(p, axis = -1)

print('{:10} {:15}'.format('word', 'predicted NER'))
print('-' * 50)
for w, pred in zip(new_sentence, p[0]):
    print('{:10} {:15}'.format(w, index_to_ner[pred]))
