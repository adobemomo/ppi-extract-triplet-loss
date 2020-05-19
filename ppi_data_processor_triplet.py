import pickle
import random
from utils import *
import logger
import time
import sys
import os
sys.stdout = logger.Logger('logs/'+time.strftime('%Y-%m-%d %H.%M.%S',time.localtime(time.time()))+os.path.basename(__file__)+'.output', sys.stdout)
sys.stderr = logger.Logger('logs/'+time.strftime('%Y-%m-%d %H.%M.%S',time.localtime(time.time()))+os.path.basename(__file__)+'.error', sys.stderr)		# redirect std err, if necessary


# step1_data: # step1_data: [sent_id, sent_text, [entity1, entity2, ppi]]
# neg_samples: [sent_id, replaced_sent, p1, p1_type, p2, p2_type, ppi]

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
            p1, p1_type, p1_offset = p[0]
            p2, p2_type, p2_offset = p[1]
            if p1 not in e_dict:
                e_dict.append(p1_offset)
            if p2 not in e_dict:
                e_dict.append(p2_offset)

        for p in pair:
            ppi = p[2]
            if ppi == 'false':
                datum = samplize_ppi(sid, sent, p, e_dict)
                if datum is None:
                    pass
                else:
                    homo_negs.append(datum)

        for p in pair:
            p1, p1_type, p1_offset = p[0]
            p2, p2_type, p2_offset = p[1]
            ppi = p[2]
            if p1 == p2:
                continue

            # generate triplet samples
            # <pos, homo_neg, non_homo_neg>
            if ppi == 'true' and len(homo_negs) > 0:
                # generate pos
                pos = samplize_ppi(sid, sent, p, e_dict)
                if pos is None:
                    continue
                # generate homo_neg
                index = random.randint(0, len(homo_negs)-1)
                # if sid == 'AIMed.d29.s249':
                #     print(homo_negs)
                #     print(index)
                homo_neg = homo_negs[index]
                # generate non_homo_neg
                for i in range(100):
                    index2 = random.randint(0,len(neg_samples)-1)
                    candidate_sample = neg_samples[index2]
                    if candidate_sample[0] != sid:
                        non_homo_neg = candidate_sample

                # if pos is None or homo_neg is None or non_homo_neg is None:
                #     continue

                triplets.append([pos, homo_neg, non_homo_neg])
                pos_instances.append(pos)
                homo_neg_instances.append(homo_neg)
                non_homo_neg_instances.append(non_homo_neg)

                # print(pos, '+', homo_neg, '+',  non_homo_neg, '=====')

    print("triplets:", len(triplets))
    return triplets, pos_instances, homo_neg_instances, non_homo_neg_instances


neg_samples_pickle = 'dataset/ppi_data/step2/neg_samples.pickle'
neg_samples = pickle.load(open(neg_samples_pickle, 'rb'))
# print(neg_samples)
print("number of neg samples: ", len(neg_samples))
ppi_step1_pickle = 'dataset/ppi_data/step1/train.pickle'
step1_data = pickle.load(open(ppi_step1_pickle, 'rb'))
triplets, pos_instances, homo_neg_instances, non_homo_neg_instances = generate_triplet(step1_data, neg_samples)
triplets_txt = 'dataset/ppi_data/triplet/triplets.txt'
triplets_pickle = 'dataset/ppi_data/triplet/triplets.pickle'
pos_instances_txt = 'dataset/ppi_data/triplet/pos_instances.txt'
pos_instances_pickle = 'dataset/ppi_data/triplet/pos_instances.pickle'
homo_neg_instances_txt = 'dataset/ppi_data/triplet/homo_neg_instances.txt'
homo_neg_instances_pickle = 'dataset/ppi_data/triplet/homo_neg_instances.pickle'
non_homo_neg_instances_txt = 'dataset/ppi_data/triplet/non_homo_neg_instances.txt'
non_homo_neg_instances_pickle = 'dataset/ppi_data/triplet/non_homo_neg_instances.pickle'


print("number of pos_instances: ", len(pos_instances))
print("number of homo_neg_instances: ", len(homo_neg_instances))
print("number of non_homo_neg_instances: ", len(non_homo_neg_instances))

write_triplets_as_txt(triplets, triplets_txt)
write_step2_data_as_txt(pos_instances,pos_instances_txt)
write_step2_data_as_txt(homo_neg_instances, homo_neg_instances_txt)
write_step2_data_as_txt(non_homo_neg_instances, non_homo_neg_instances_txt)

pickle.dump(triplets, open(triplets_pickle, 'wb'))
pickle.dump(pos_instances, open(pos_instances_pickle, 'wb'))
pickle.dump(homo_neg_instances, open(homo_neg_instances_pickle, 'wb'))
pickle.dump(non_homo_neg_instances, open(non_homo_neg_instances_pickle, 'wb'))

