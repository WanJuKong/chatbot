from konlpy.tag import Komoran
import pickle

class Preproceses:
    def __init__(self, word2index_dic = '', userdic = None):
        if(word2index_dic != ''):
            f = open(word2index_dic, 'rb')
            self.word_index = pickle.load(f)
            f.close()
        else:
            self.word_index = None

    def pos(self, sentence):
        return self.komoran.pos(sentence)

    def get_keywords(self, pos, without_tag = False):
        f = lambda x: x in self.exclusion_tags
        word_list = []
        for p in pos:
            if f(p[1]) is False:
                word_list.append(p if without_tag is False else p[0])
        return word_list

    def get_wordidx_sequence(self, kewords):
        if self.word_index is None:
            return []
        w2i = []
        for word in keywords:
            try:
                w2i.append(self.word_index[word])
            except KeyError:
                w2i.append(self.word_index['OOV'])
        return w2i

