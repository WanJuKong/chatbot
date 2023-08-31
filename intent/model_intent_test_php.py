# 8_9

import sys
from Preprocess import Preprocess
from IntentModel import IntentModel

p = Preprocess('./data/chatbot_dict.bin', './data/user_dic.tsv')
intent = IntentModel('./models/intent_model.keras', p)

query = sys.argv[1]
predict = intent.predict_class(query)
predict_label = intent.labels[predict]
print(query)
print('class: ', predict)
print('lable: ', predict_label)
