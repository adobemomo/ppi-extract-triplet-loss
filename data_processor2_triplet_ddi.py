import pickle
import random
from utils import *

# step1_data: # step1_data: [sent_id, sent_text, [entity1, entity2, ddi]]
# neg_samples: [sent_id, replaced_sent, d1, d1_type, d2, d2_type, ddi]

def generate_triplet(step1_data, neg_samples):
    triplets = []
    pos_instances = []
    homo_neg_instances = []
    non_homo_neg_instances = []
    for s in step1_data:
        sid = s[0]
        sent = s[1]
        pair = s[2]

        e_dict = []
        homo_negs = []

        for p in pair:
            d1, d1_type, d1_offset = p[0]
            d2, d2_type, d2_offset = p[1]
            if d1 not in e_dict:
                e_dict.append(d1_offset)
            if d2 not in e_dict:
                e_dict.append(d2_offset)

        for p in pair:
            ddi = p[2]
            if ddi == 'false':
                homo_negs.append(samplize(sid, sent, p, e_dict))

        for p in pair:
            d1, d1_type, d1_offset = p[0]
            d2, d2_type, d2_offset = p[1]
            ddi = p[2]
            if d1 == d2:
                continue

            # generate triplet samples
            # <pos, homo_neg, non_homo_neg>
            if ddi == 'true' and len(homo_negs) > 0:
                # generate pos
                pos = samplize(sid, sent, p, e_dict)
                # generate homo_neg
                index = random.randint(0, len(homo_negs)-1)
                homo_neg = homo_negs[index]
                # generate non_homo_neg
                for i in range(100):
                    index2 = random.randint(0,len(neg_samples)-1)
                    candidate_sample = neg_samples[index2]
                    if candidate_sample[0] != sid:
                        non_homo_neg = candidate_sample

                triplets.append([pos, homo_neg, non_homo_neg])
                pos_instances.append(pos)
                homo_neg_instances.append(homo_neg)
                non_homo_neg_instances.append(non_homo_neg)

                # print(pos, '+', homo_neg, '+',  non_homo_neg, '=====')

    return triplets, pos_instances, homo_neg_instances, non_homo_neg_instances


neg_samples_pickle = 'dataset/ddi_data/step2/neg_samples.pickle'
neg_samples = pickle.load(open(neg_samples_pickle, 'rb'))
# print(neg_samples)
# print("number of neg samples: ", len(neg_samples))
ddi_step1_pickle = 'dataset/ddi_data/step1/train.pickle'
step1_data = pickle.load(open(ddi_step1_pickle, 'rb'))
triplets, pos_instances, homo_neg_instances, non_homo_neg_instances = generate_triplet(step1_data, neg_samples)
triplets_txt = 'dataset/ddi_data/triplet/triplets.txt'
triplets_pickle = 'dataset/ddi_data/triplet/triplets.pickle'
pos_instances_txt = 'dataset/ddi_data/triplet/pos_instances.txt'
pos_instances_pickle = 'dataset/ddi_data/triplet/pos_instances.pickle'
homo_neg_instances_txt = 'dataset/ddi_data/triplet/homo_neg_instances.txt'
homo_neg_instances_pickle = 'dataset/ddi_data/triplet/homo_neg_instances.pickle'
non_homo_neg_instances_txt = 'dataset/ddi_data/triplet/non_homo_neg_instances.txt'
non_homo_neg_instances_pickle = 'dataset/ddi_data/triplet/non_homo_neg_instances.pickle'

write_triplets_as_txt(triplets, triplets_txt)
write_step2_data_as_txt(pos_instances,pos_instances_txt)
write_step2_data_as_txt(homo_neg_instances, homo_neg_instances_txt)
write_step2_data_as_txt(non_homo_neg_instances, non_homo_neg_instances_txt)

pickle.dump(triplets, open(triplets_pickle, 'wb'))
pickle.dump(pos_instances, open(pos_instances_pickle, 'wb'))
pickle.dump(homo_neg_instances, open(homo_neg_instances_pickle, 'wb'))
pickle.dump(non_homo_neg_instances, open(non_homo_neg_instances_pickle, 'wb'))

