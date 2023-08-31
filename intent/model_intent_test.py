# 8_9

from Preprocess import Preprocess
from IntentModel import IntentModel

p = Preprocess('./data/chatbot_dict.bin', './data/user_dic.tsv')
intent = IntentModel('./models/intent_model.keras', p)

query = input(':')
while query != 'e':
    predict = intent.predict_class(query)
    predict_label = intent.labels[predict]
    print(query)
    print(intent.class_probabilities(query))
    print('class: {}\nlable: {}\n\n'.format(predict, predict_label))
    query = input(':')
