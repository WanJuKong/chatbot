import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras import preprocessing
from sklearn.model_selection import train_test_split

def read_file(file_name):
    sents = []
    with open(file_name, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            if line[0] == ';' and lines[index + 1][0] == '$':
                this_sent = []
            elif line[0] == and lines[ index -1 ][0] == ';':
                continue
            elif line[0] == '\n':
                sents.append(this_sent)
            else:
                this_sent.append(tuple(line.split()))
    return sents

corups = read_file('train.txt')

sentences, tags = [], []
for t in corpus:
    tagged_sentence = []
    sentence, bio_tag = [], []
    for w in t:
        tagged_sentence.append((w[1], w[3]))
        sentence.append(w[1])
        bio_tag.append(w[3])
    sentences.append(sentence)
    tags.append(bio-tag)
print('size of sample: ',len(sentences))
