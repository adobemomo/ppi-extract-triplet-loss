# replace ddi with DRUGA, DRUGB and DRUGN
# save all negative samples
import pickle
from utils import *


# data: [sent_id, sent_text, pair_list]
# pair_list: [entity1, entity2, ddi]
# ddi: true/false
# samplized_data: [sent_id, replaced_sent, d1, d1_type, d2, d2_type, ddi]
def samplize(data):
    count = 0
    samplized_data = []
    for s in data:
        id = s[0]
        sent = s[1]
        pair = s[2]

        e_dict = []

        for p in pair:
            d1, d1_type, d1_offset = p[0]
            d2, d2_type, d2_offset = p[1]
            if d1 not in e_dict:
                e_dict.append(d1_offset)
            if d2 not in e_dict:
                e_dict.append(d2_offset)

        for p in pair:
            d1, d1_type, d1_offset = p[0]
            d2, d2_type, d2_offset = p[1]
            ddi = p[2]
            if d1 == d2:
                continue
            count += 1

            if d1_offset.find(';') == -1 and d2_offset.find(';') == -1:
                d1_start, d1_end = d1_offset.split('-')
                d2_start, d2_end = d2_offset.split('-')
                d1_start, d1_end = int(d1_start), int(d1_end)
                d2_start, d2_end = int(d2_start), int(d2_end)
            elif d1_offset.find(';') > -1 and d2_offset.find(';') > -1:
                d1_start, d1_end = d1_offset.split(';')[0].split('-')
                d2_start, d2_end = d2_offset.split(';')[0].split('-')
                d1_start, d1_end = int(d1_start), int(d1_end)
                d2_start, d2_end = int(d2_start), int(d2_end)
            elif d1_offset.find(';') > -1 and d2_offset.find(';') == -1:
                d1_1, d1_2 = d1_offset.split(';')
                d1_1_start, d1_1_end = d1_1.split('-')
                d1_2_start, d1_2_end = d1_2.split('-')
                d1_1_start, d1_1_end = int(d1_1_start), int(d1_1_end)
                d1_2_start, d1_2_end = int(d1_2_start), int(d1_2_end)
                d2_start, d2_end = d2_offset.split('-')
                d2_start, d2_end = int(d2_start), int(d2_end)

                if len(set(range(d1_1_start, d1_1_end)) & set(range(d2_start, d2_end))):
                    d1_start, d1_end = d1_2_start, d1_2_end
                else:
                    d1_start, d1_end = d1_1_start, d1_1_end
            else:
                d2_1, d2_2 = d2_offset.split(';')
                d2_1_start, d2_1_end = d2_1.split('-')
                d2_2_start, d2_2_end = d2_2.split('-')
                d2_1_start, d2_1_end = int(d2_1_start), int(d2_1_end)
                d2_2_start, d2_2_end = int(d2_2_start), int(d2_2_end)
                d1_start, d1_end = d1_offset.split('-')
                d1_start, d1_end = int(d1_start), int(d1_end)

                if len(set(range(d2_1_start, d2_1_end)) & set(range(d1_start, d1_end))):
                    d2_start, d2_end = d2_2_start, d2_2_end
                else:
                    d2_start, d2_end = d2_1_start, d2_1_end

            other = set(e_dict) - set([d1_offset, d2_offset])
            replaced_sent = sent.replace(sent[d1_start:d1_end + 1], 'DRUGA').replace(sent[d2_start:d2_end + 1], 'DRUGB')
            for n in other:
                if n.find(';') > -1:
                    n = n.split(';')[0]
                n_start, n_end = n.split('-')
                n_start, n_end = int(n_start), int(n_end)
                replaced_sent = replaced_sent.replace(sent[n_start:n_end + 1], 'DRUGN')

            # print(replaced_sent)
            # print(sent)
            samplized_data.append([id, replaced_sent, d1, d1_type, d2, d2_type, ddi])

    # print(samplized_data)
    return samplized_data

# data: [sent_id, replaced_sent, d1, d1_type, d2, d2_type, ddi]
# neg_samples: ddi == 'false'
def collect_neg_samples(data):
    neg_samples = []
    for d in data:
        if d[6] == 'false':
            neg_samples.append(d)
    return neg_samples


ddi_step1_pickle = 'dataset/ddi_data/step1/train.pickle'
ddi_step2_samples_pickle = 'dataset/ddi_data/step2/all_samples.pickle'
ddi_step2_samples_txt = 'dataset/ddi_data/step2/all_samples.txt'
ddi_step2_neg_samples_pickle = 'dataset/ddi_data/step2/neg_samples.pickle'
ddi_step2_neg_samples_txt = 'dataset/ddi_data/step2/neg_samples.txt'
step1_data = pickle.load(open(ddi_step1_pickle, 'rb'))
samplized_data = samplize(step1_data)
neg_samples = collect_neg_samples(samplized_data)
# print(samplized_data)
# print("samplized_data number: ", len(samplized_data))
# print(neg_samples)
# print("neg_samples number: ", len(neg_samples))
write_step2_data_as_txt(samplized_data, ddi_step2_samples_txt)
write_step2_data_as_txt(neg_samples, ddi_step2_neg_samples_txt)
pickle.dump(samplized_data, open(ddi_step2_samples_pickle, 'wb'))
pickle.dump(neg_samples, open(ddi_step2_neg_samples_pickle, 'wb'))