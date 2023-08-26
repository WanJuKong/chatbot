from Preprocess import Preprocess
from IntentModel import IntentModel

p = Preprocess('chatbot_dict.bin', 'user_dic.tsv')
intent = IntentModel('./models/intent_model.keras', p)
query = input(':')

predict = intent.predict_class(query)
predict_label = intent.labels[predict]

print(query)

print('class: ', predict)
print('lable: ', predict_label)
