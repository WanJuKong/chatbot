import csv

data = []

with open('./data/total_train_data_raw.csv' , 'r') as f:
    csv_reader = csv.DictReader(f)
    for row in csv_reader:
        intent = int(row['intent'])
        if intent % 2 == 1: pass
        else:
            row['intent'] = int(intent/2)
            data.append(row)

fieldnames = ['query', 'intent']

with open('./data/total_train_data.csv', 'w', newline='') as f:
    csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
    csv_writer.writeheader()
    for row in data:
        csv_writer.writerow(row)
