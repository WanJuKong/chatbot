import pickle
from Preprocess import Preprocess
f = open('chatbot_dict.bin', 'rb')
word_index = pickle.load(f)
f.close()

sent = input(':')

p = Preprocess(userdic = './data/userdic.tsv')

pos=p.pos(sent)

keywords = p.get_keywords(pos, without_tag = True)
for word in keywords:
    try:
        print(word, word_index[word])
    except KeyError:
        print(word, word_index['OOV'])
