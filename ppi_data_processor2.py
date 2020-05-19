# replace ppi with PROTEINA, PROTEINB and PROTEINN
# save all negative samples
import pickle
from utils import *
import logger
import time
import sys
import os
sys.stdout = logger.Logger('logs/'+time.strftime('%Y-%m-%d %H.%M.%S',time.localtime(time.time()))+os.path.basename(__file__)+'.output', sys.stdout)
sys.stderr = logger.Logger('logs/'+time.strftime('%Y-%m-%d %H.%M.%S',time.localtime(time.time()))+os.path.basename(__file__)+'.error', sys.stderr)		# redirect std err, if necessary


# data: [sent_id, sent_text, pair_list]
# pair_list: [entity1, entity2, ppi]
# ppi: true/false
# samplized_data: [sent_id, replaced_sent, p1, p1_type, p2, p2_type, ppi]
def samplize(data):
    count = 0
    samplized_data = []
    for s in data:
        id = s[0]
        sent = s[1]
        pair = s[2]

        e_dict = []

        for p in pair:
            p1, p1_type, p1_offset = p[0]
            p2, p2_type, p2_offset = p[1]
            if p1 not in e_dict:
                e_dict.append(p1_offset)
            if p2 not in e_dict:
                e_dict.append(p2_offset)

        for p in pair:
            p1, p1_type, p1_offset = p[0]
            p2, p2_type, p2_offset = p[1]

            datum = samplize_ppi(id, sent, p, e_dict)
            if datum is None:
                pass
            else:
                samplized_data.append(datum)

    return samplized_data

# data: [sent_id, replaced_sent, p1, p1_type, p2, p2_type, ppi]
# neg_samples: ppi == 'false'
def collect_neg_samples(data):
    neg_samples = []
    pos_samples = []
    for d in data:
        if d[6] == 'false':
            neg_samples.append(d)
        else:
            pos_samples.append(d)
    #
    # neg_samples = neg_samples[:int(len(neg_samples)/30)]
    # pos_samples = pos_samples[:int(len(pos_samples)/30)]
    return neg_samples, pos_samples


def generate_dataset(negdata, posdata):
    train = []
    test = []
    val = []
    cnt = 0
    for datum in posdata:
        cnt += 1
        cnt %= 5
        if cnt == 1:
            test.append(datum)
        elif cnt == 2:
            val.append(datum)
        else:
            train.append(datum)
    for datum in negdata:
        cnt += 1
        cnt %= 5
        if cnt == 1:
            test.append(datum)
        elif cnt == 2:
            val.append(datum)
        else:
            train.append(datum)
    return train, test, val



ppi_step1_pickle = 'dataset/ppi_data/step1/train.pickle'
ppi_step2_samples_pickle = 'dataset/ppi_data/step2/all_samples.pickle'
ppi_step2_samples_txt = 'dataset/ppi_data/step2/all_samples.txt'
ppi_step2_neg_samples_pickle = 'dataset/ppi_data/step2/neg_samples.pickle'
ppi_step2_neg_samples_txt = 'dataset/ppi_data/step2/neg_samples.txt'
ppi_step2_pos_samples_pickle = 'dataset/ppi_data/step2/pos_samples.pickle'
ppi_step2_pos_samples_txt = 'dataset/ppi_data/step2/pos_samples.txt'
ppi_train_pickle = 'dataset/ppi_train.pickle'
ppi_train_txt = 'dataset/ppi_train.txt'
ppi_test_pickle = 'dataset/ppi_test.pickle'
ppi_test_txt = 'dataset/ppi_test.txt'
ppi_val_pickle = 'dataset/ppi_val.pickle'
ppi_val_txt = 'dataset/ppi_val.txt'


step1_data = pickle.load(open(ppi_step1_pickle, 'rb'))
samplized_data = samplize(step1_data)
neg_samples, pos_samples = collect_neg_samples(samplized_data)
train, test, val = generate_dataset(neg_samples, pos_samples)
# print(samplized_data)
print("samplized_data number: ", len(samplized_data))
# print(neg_samples)
print("neg_samples number: ", len(neg_samples))
print("pos_samples number: ", len(pos_samples))
print("train size :", len(train))
print("test size :", len(test))
print("val size :", len(val))

write_step2_data_as_txt(samplized_data, ppi_step2_samples_txt)
write_step2_data_as_txt(neg_samples, ppi_step2_neg_samples_txt)
write_step2_data_as_txt(pos_samples, ppi_step2_pos_samples_txt)
write_step2_data_as_txt(train, ppi_train_txt)
write_step2_data_as_txt(test, ppi_test_txt)
write_step2_data_as_txt(val, ppi_val_txt)

pickle.dump(samplized_data, open(ppi_step2_samples_pickle, 'wb'))
pickle.dump(neg_samples, open(ppi_step2_neg_samples_pickle, 'wb'))
pickle.dump(pos_samples, open(ppi_step2_pos_samples_pickle, 'wb'))
pickle.dump(train, open(ppi_train_pickle, 'wb'))
pickle.dump(test, open(ppi_test_pickle, 'wb'))
pickle.dump(val, open(ppi_val_pickle, 'wb'))

