from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from rnn import *
import numpy as np
import sklearn as sk
import random
import csv
import re
import collections
import pickle
import sys
import os
sys.path.append("source")
from utils import *
import logger
import time
import sys
import os
sys.stdout = logger.Logger('logs/'+time.strftime('%Y-%m-%d %H.%M.%S',time.localtime(time.time()))+os.path.basename(__file__)+'.output', sys.stdout)
sys.stderr = logger.Logger('logs/'+time.strftime('%Y-%m-%d %H.%M.%S',time.localtime(time.time()))+os.path.basename(__file__)+'.error', sys.stderr)		# redirect std err, if necessary


embeSize = 100
dist1_emb_size = 10
dist2_emb_size = 10
type_emb_size = 10
numfilter = 200

result_file = 'results/result.txt'
sent_out = 'results/multi_sents_'

num_epochs = 25
check_point = [5, 10, 15, 20, 25]
batch_size = 200
reg_para = 0.001
drop_out = 1.0

# file_train = "dataset/train_data.txt"
# file_val = "dataset/val_data.txt"
# file_test = "dataset/test_data.txt"

pickle_train = 'dataset/ppi_train.pickle'
pickle_test = 'dataset/ppi_test.pickle'
pickle_val = 'dataset/ppi_val.pickle'

word2vec_emb_file = "word2Vec.vector"


# read train dataset
# train_sent_contents, train_e1_list, train_e2_list, train_label_lists = dataRead(file_train)
train_sent_contents, train_e1_list, train_e2_list, train_label_lists = read_pickle(pickle_train)
# add position information
train_token_list, train_dist1_list, train_dist2_list, train_type_list = makeFeatures(train_sent_contents,
                                                                                     train_e1_list,
                                                                                     train_e2_list)
# read val dataset
# val_sent_contents, val_e1_list, val_e2_list, val_label_lists = dataRead(file_val)
val_sent_contents, val_e1_list, val_e2_list, val_label_lists = read_pickle(pickle_val)
# add position information
val_token_list, val_dist1_list, val_dist2_list, val_type_list = makeFeatures(val_sent_contents,val_e1_list,val_e2_list)

# read test dataset
# test_sent_contents, test_e1_list, test_e2_list, test_label_lists = dataRead(file_test)
test_sent_contents, test_e1_list, test_e2_list, test_label_lists = read_pickle(pickle_test)
test_token_list, test_dist1_list, test_dist2_list, test_type_list = makeFeatures(test_sent_contents,test_e1_list,test_e2_list)

# add position information
print("train_size:", len(train_token_list))
print("val_size:", len(val_token_list))
print("test_size:", len(test_token_list))


train_sent_lengths, val_sent_lengths, test_sent_lengths = findSentLengths([train_token_list, val_token_list, test_token_list])
sentMax = max(train_sent_lengths + val_sent_lengths + test_sent_lengths)
print("max sentence length:", sentMax)
train_sent_lengths = np.array(train_sent_lengths, dtype='int32')
val_sent_lengths = np.array(val_sent_lengths, dtype='int32')
test_sent_lengths = np.array(test_sent_lengths, dtype='int32')


# create dictionary for labels, tokens and distance
# label_dict = {'false':0, 'advise': 1, 'mechanism': 2, 'effect': 3, 'int': 4}
label_dict = {'false':0, 'true':1}
token_dict = makeWordList([train_token_list,val_token_list,test_token_list])
dist1_dict = makeDistanceList([train_dist1_list, val_dist1_list, test_dist1_list])
dist2_dict = makeDistanceList([train_dist2_list, val_dist2_list, test_dist2_list])
type_dict = makeDistanceList([train_type_list, val_type_list, test_type_list])
print("Word dictionary length:", len(token_dict))


# Word embedding
word_vector_pickle = 'word_vector.pickle'

# wv = readWordEmb(token_dict, word2vec_emb_file, embeSize)
# pickle.dump(wv, open(word_vector_pickle, 'wb'))
print("Loading word vector")

wv = pickle.load(open(word_vector_pickle, 'rb'))

# Mapping Train
print("Mapping train dataset")
token_train = mapWordToId(train_token_list,token_dict)
dist1_train = mapWordToId(train_dist1_list,dist1_dict)
dist2_train = mapWordToId(train_dist2_list,dist2_dict)
type_train = mapWordToId(train_type_list, type_dict)
label_t = mapLabelToId(train_label_lists,label_dict)
# transform label_train into one hot
label_train = np.zeros((len(label_t),len(label_dict)))
for i in range(len(label_t)):
    label_train[i][label_t[i]] = 1.0

#mapping validation
token_val = mapWordToId(val_token_list, token_dict)
dist1_val = mapWordToId(val_dist1_list,dist1_dict)
dist2_val = mapWordToId(val_dist2_list,dist2_dict)
type_val = mapWordToId(val_type_list, type_dict)
label_v = mapLabelToId(val_label_lists, label_dict)
# transform label_train into one hot
label_val = np.zeros((len(label_v), len(label_dict)))
for i in range(len(label_v)):
    label_val[i][label_v[i]] = 1.0

# mapping test
token_test = mapWordToId(test_token_list, token_dict)
dist1_test = mapWordToId(test_dist1_list, dist1_dict)
dist2_test = mapWordToId(test_dist2_list, dist2_dict)
type_test = mapWordToId(test_type_list, type_dict)
label_t = mapLabelToId(test_label_lists, label_dict)
# transform label_train into one hot
label_test = np.zeros((len(label_t), len(label_dict)))
for i in range(len(label_t)):
    label_test[i][label_t[i]] = 1.0



# padding
token_train, dist1_train, dist2_train, type_train = paddData([token_train, dist1_train, dist2_train, type_train],sentMax)
token_val, dist1_val, dist2_val, type_val = paddData([token_val, dist1_val, dist2_val, type_val],sentMax)
token_test, dist1_test, dist2_test, type_test = paddData([token_test, dist1_test, dist2_test, type_test],sentMax)

# if not os.path.isfile('data_pickle'):
#     with open('data_pickle', 'wb') as handle:
#         pickle.dump(token_train, handle)
#         pickle.dump(dist1_train, handle)
#         pickle.dump(dist2_train, handle)
#         pickle.dump(type_train, handle)
#         pickle.dump(label_train, handle)
#         pickle.dump(train_sent_lengths, handle)
#
#         pickle.dump(token_val, handle)
#         pickle.dump(dist1_val, handle)
#         pickle.dump(dist2_val, handle)
#         pickle.dump(type_val, handle)
#         pickle.dump(label_val, handle)
#         pickle.dump(val_sent_lengths, handle)
#
#         pickle.dump(token_test, handle)
#         pickle.dump(dist1_test, handle)
#         pickle.dump(dist2_test, handle)
#         pickle.dump(type_test, handle)
#         pickle.dump(label_test, handle)
#         pickle.dump(test_sent_lengths, handle)
#
# else:
#     with open('data_pickle', 'rb') as handle:
#         token_train = pickle.load(handle)
#         dist1_train = pickle.load(handle)
#         dist2_train = pickle.load(handle)
#         type_train = pickle.load(handle)
#         label_train = pickle.load(handle)
#         train_sent_lengths = pickle.load(handle)
#
#         token_val = pickle.load(handle)
#         dist1_val = pickle.load(handle)
#         dist2_val = pickle.load(handle)
#         type_val = pickle.load(handle)
#         label_val = pickle.load(handle)
#         val_sent_lengths = pickle.load(handle)
#
#         token_test = pickle.load(handle)
#         dist1_test = pickle.load(handle)
#         dist2_test = pickle.load(handle)
#         type_test = pickle.load(handle)
#         label_test = pickle.load(handle)
#         test_sent_lengths = pickle.load(handle)


# vocabulary size
token_dict_size = len(token_dict)
dist1_dict_size = len(dist1_dict)
dist2_dict_size = len(dist2_dict)
type_dict_size = len(type_dict)
label_dict_size = len(label_dict)
print("token_dict_size:", token_dict_size)
print("dist1_dict_size:", dist1_dict_size)
print("dist2_dict_size:", dist2_dict_size)
print("type_dict_size:", type_dict_size)
print("label_dict_size:", label_dict_size)

rev_token_dict = makeWordListReverst(token_dict)
rev_label_dict = {0:'false', 1:'true'}
# rev_label_dict = {0:'false', 1:'advise', 2:'mechanism', 3:'effect', 4:'int'}

fp = open(result_file, 'a+')  # keep precision recall
fsent = open(sent_out, 'w') # keep sentence and its results

rnn = RNN_Relation(label_dict_size,  # output layer size
                   token_dict_size,  # word embedding size
                   dist1_dict_size,  # position embedding size
                   dist2_dict_size,  # position embedding size
                   type_dict_size,  # type embedding size
                   sentMax,  # length of sentence
                   wv,  # word embedding
                   d1_emb_size = dist1_emb_size,  # length of position embedding
                   d2_emb_size = dist2_emb_size,  # length of position embedding
                   type_emb_size = type_emb_size,
                   num_filters = numfilter,  # number of hidden nodes in RNN
                   w_emb_size = embeSize,
                   l2_reg_lambda = reg_para  # l2 reg
                   )

train_len = len(token_train)

loss_list = []
test_res = []
val_res = []
fscore_val = []
fscore_test = []

def test_step(W, sent_lengths, d1, d2, type, label):
    n = len(W)
    print("num_W:", len(W))

    num_batch = int(n/batch_size)
    samples = []
    print(range(num_batch))
    for i in range(num_batch):
        print(i)
        print(max(batch_size*(i+1), n))
        samples.append(range(batch_size*i, batch_size*(i+1)))
    samples.append(range(batch_size * num_batch,  n))

    prediction = []
    print(samples)
    for i in samples:
        p, a = rnn.test_step(W[i], sent_lengths[i], d1[i], d2[i], type[i], label[i])
        prediction.extend(p)
    return prediction


num_batches_per_epoch = int(train_len/batch_size) + 1
iii = 0  # checkpoint number
for epoch in range(num_epochs):
    shuffle_indices = np.random.permutation(np.arange(train_len))
    token_tr = token_train[shuffle_indices]
    dist1_tr = dist1_train[shuffle_indices]
    dist2_tr = dist2_train[shuffle_indices]
    type_tr = type_train[shuffle_indices]
    label_tr = label_train[shuffle_indices]
    sent_tr = train_sent_lengths[shuffle_indices]
    loss_epoch = 0.0

    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num+1)*batch_size, train_len)
        loss = rnn.train_step(token_tr[start_index:end_index], sent_tr[start_index:end_index], dist1_tr[start_index:end_index],
                              dist2_tr[start_index:end_index], type_tr[start_index:end_index], label_tr[start_index:end_index],
                              drop_out)
        loss_epoch += loss

    print("Epoch_:", epoch, " loss:,", loss_epoch)
    loss_list.append(round(loss_epoch, 5))

    if epoch in check_point:
    # if epoch == 1:
        iii += 1
        #
        # saver = tf.train.Saver()
        # path = saver.save(rnn.sess, 'saved_models/model_'+str(iii)+'.ckpt')

        # Validation
        y_pred_val = test_step(token_val, val_sent_lengths, dist1_val, dist2_val, type_val, label_val)
        y_true_val = np.argmax(label_val, 1)
        # print("num_token_val:", len(token_val))
        # print("y_pred_val_shape:", len(y_pred_val))
        # print("y_true_val_shape:", len(y_true_val))
        fscore_val.append(f1_score(y_true_val, y_pred_val, [1], average='micro'))
        val_res.append([y_true_val, y_pred_val])
        print("fscore_val:", fscore_val)

        # Test
        y_pred_test = test_step(token_test, test_sent_lengths, dist1_test, dist2_test, type_test, label_test)
        y_true_test = np.argmax(label_test, 1)
        # print("y_pred_test_shape:",y_pred_test.get_shape())
        # print("y_true_test_shape:", y_true_test.get_shape())
        fscore_test.append(f1_score(y_true_test, y_pred_test, [1], average='micro'))
        test_res.append([y_true_test, y_pred_test])
        print("fscore_val:", fscore_val)

print("Train over.")

index = np.argmax(fscore_val)    # Best epoch from validation set
y_true, y_pred = test_res[index]   # actual prediction


# fp.write('\n Results in Test Set (Best Index) '+str(ind)+'\n')
# fp.write(str(precision_score(y_true, y_pred,[1,2,3,4], average='micro' )))
# fp.write('\t')
# fp.write(str(recall_score(y_true, y_pred, [1,2,3,4], average='micro' )))
# fp.write('\t')
# fp.write(str(f1_score(y_true, y_pred, [1,2,3,4], average='micro' )))
# fp.write('\t')
# fp.write('\n')

print("Start writing res.")

fp.write('\n Results in Test Set (Best Index )' + str(index)+'\n')
fp.write(str(precision_score(y_true, y_pred, [1], average='micro')))
fp.write('\t')
fp.write(str(recall_score(y_true, y_pred, [1], average='micro' )))
fp.write('\t')
fp.write(str(f1_score(y_true, y_pred, [1], average='micro' )))
fp.write('\t')
fp.write('\n')

# fp.write('class 2\t')
# fp.write(str(precision_score(y_true, y_pred,[2], average='micro' )))
# fp.write('\t')
# fp.write(str(recall_score(y_true, y_pred, [2], average='micro' )))
# fp.write('\t')
# fp.write(str(f1_score(y_true, y_pred, [2], average='micro' )))
# fp.write('\n')
#
# fp.write('class 3\t')
# fp.write(str(precision_score(y_true, y_pred,[3], average='micro' )))
# fp.write('\t')
# fp.write(str(recall_score(y_true, y_pred, [3], average='micro' )))
# fp.write('\t')
# fp.write(str(f1_score(y_true, y_pred, [3], average='micro' )))
# fp.write('\n')
#
# fp.write('class 4\t')
# fp.write(str(precision_score(y_true, y_pred,[4], average='micro' )))
# fp.write('\t')
# fp.write(str(recall_score(y_true, y_pred, [4], average='micro' )))
# fp.write('\t')
# fp.write(str(f1_score(y_true, y_pred, [4], average='micro' )))
# fp.write('\n')

fp.write(str(confusion_matrix(y_true, y_pred)))
fp.write('\n')

for sent, slen, y_t, y_p, in zip(token_test, test_sent_lengths, y_true, y_pred):
    sent_1 = [str(rev_token_dict[sent[kk]]) for kk in range(slen)]
    s = ' '.join(sent_1)
    fsent.write(s)
    fsent.write('\n')
    fsent.write(rev_label_dict[y_t])
    fsent.write('\n')
    fsent.write(rev_label_dict[y_p])
    fsent.write('\n')
    fsent.write('\n')
fsent.close()
rnn.sess.close()

print("All over")








