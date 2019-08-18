import numpy as np

import csv, pickle

from model import TransE

TRAIN_DATASET_PATH = './FB15k/freebase_mtr100_mte100-train.txt'

def load(file_path):
    with open(TRAIN_DATASET_PATH, 'r') as f:
        train_tsv_reader = csv.reader(f, delimiter='\t')
        data = []
        entity_dic = {}
        link_dic = {}
        entity_idx = 0
        link_idx = 0
        for row in train_tsv_reader: 
            head = row[0]
            link = row[1]
            tail = row[2]
            if not head in entity_dic:
                entity_dic[head] = entity_idx
                entity_idx += 1

            if not tail in entity_dic:
                entity_dic[tail] = entity_idx
                entity_idx += 1

            if not link in link_dic:
                link_dic[link] = link_idx
                link_idx += 1
            data.append((entity_dic[head], link_dic[link], entity_dic[tail]))
        return data, entity_dic, link_dic

train_data, entity_dic, link_dic = load(TRAIN_DATASET_PATH)

#training
model = TransE(len(entity_dic), len(link_dic), 1, 50, 0.01, 50)
model.fit(np.array(train_data))

with open('models/transe.pkl', 'wb') as f:
    pickle.dump(model, f)
