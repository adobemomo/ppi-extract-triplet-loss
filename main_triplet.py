from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from model_triplet import *

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

result_file = 'results_triplet/result.txt'
sent_out = 'results_triplet/multi_sents_'

num_epochs = 25
check_point = [5, 10, 15, 20, 25]
batch_size = 150
reg_para = 0.001
drop_out = 1.0

# file_train = "dataset/train_data.txt"
# file_train_pos = "dataset/ppi_data/triplet/pos_instances.txt"
# file_train_neg = "dataset/ppi_data/triplet/homo_neg_instances.txt"
# file_train_non_neg = "dataset/ppi_data/triplet/non_homo_neg_instances.txt"
# file_test = "dataset/ppi_test.txt"

pickle_train_pos = "dataset/ppi_data/triplet/pos_instances.pickle"
pickle_train_neg = "dataset/ppi_data/triplet/homo_neg_instances.pickle"
pickle_train_non_neg = "dataset/ppi_data/triplet/non_homo_neg_instances.pickle"
pickle_test = "dataset/ppi_test.pickle"

word2vec_emb_file = "word2Vec.vector"



# read train dataset
# train_pos_sent_contents, train_pos_e1_list, train_pos_e2_list, train_pos_label_lists = dataRead(file_train_pos)
# train_neg1_sent_contents, train_neg1_e1_list, train_neg1_e2_list, train_neg1_label_lists = dataRead(file_train_neg)
# train_neg2_sent_contents, train_neg2_e1_list, train_neg2_e2_list, train_neg2_label_lists = dataRead(file_train_non_neg)

train_pos_sent_contents, train_pos_e1_list, train_pos_e2_list, train_pos_label_lists = read_pickle(pickle_train_pos)
train_neg1_sent_contents, train_neg1_e1_list, train_neg1_e2_list, train_neg1_label_lists = read_pickle(pickle_train_neg)
train_neg2_sent_contents, train_neg2_e1_list, train_neg2_e2_list, train_neg2_label_lists = read_pickle(pickle_train_non_neg)

# add position information
train_pos_token_list, train_pos_dist1_list, train_pos_dist2_list, train_pos_type_list = \
    makeFeatures(train_pos_sent_contents, train_pos_e1_list, train_pos_e2_list)
train_neg1_token_list, train_neg1_dist1_list, train_neg1_dist2_list, train_neg1_type_list = \
    makeFeatures(train_neg1_sent_contents, train_neg1_e1_list, train_neg1_e2_list)
train_neg2_token_list, train_neg2_dist1_list, train_neg2_dist2_list, train_neg2_type_list = \
    makeFeatures(train_neg2_sent_contents, train_neg2_e1_list, train_neg2_e2_list)


# read test dataset
# test_sent_contents, test_e1_list, test_e2_list, test_label_lists = dataRead(file_test)
test_sent_contents, test_e1_list, test_e2_list, test_label_lists = read_pickle(pickle_test)
# add position information
test_token_list, test_dist1_list, test_dist2_list, test_type_list = makeFeatures(test_sent_contents,test_e1_list,test_e2_list)
print("train_pos_size:", len(train_pos_token_list))
print("train_neg1_size:", len(train_neg1_token_list))
print("train_neg2_size:", len(train_neg2_token_list))
print("test_size:", len(test_token_list))


train_pos_sent_lengths, train_neg1_sent_lengths, train_neg2_sent_lengths, test_sent_lengths = \
    findSentLengths([train_pos_token_list, train_neg1_token_list, train_neg2_token_list, test_token_list])
sentMax = max(train_pos_sent_lengths + train_neg1_sent_lengths + train_neg2_sent_lengths + test_sent_lengths)
print("max sentence length:", sentMax)
train_pos_sent_lengths = np.array(train_pos_sent_lengths, dtype='int32')
train_neg1_sent_lengths = np.array(train_neg1_sent_lengths, dtype='int32')
train_neg2_sent_lengths = np.array(train_neg2_sent_lengths, dtype='int32')
test_sent_lengths = np.array(test_sent_lengths, dtype='int32')


# create dictionary for labels, tokens and distance
# label_dict = {'false':0, 'advise': 1, 'mechanism': 2, 'effect': 3, 'int': 4}
label_dict = {'false':0, 'true':1}
token_dict = makeWordList([train_pos_token_list,train_neg1_token_list, train_neg2_token_list,test_token_list])
dist1_dict = makeDistanceList([train_pos_dist1_list, train_neg1_dist1_list, train_neg2_dist1_list, test_dist1_list])
dist2_dict = makeDistanceList([train_pos_dist2_list, train_neg1_dist2_list, train_neg2_dist2_list, test_dist2_list])
type_dict = makeDistanceList([train_pos_type_list, train_neg1_type_list, train_neg2_type_list, test_type_list])
print("Word dictionary length:", len(token_dict))


# Word embedding

word_vector_pickle = 'word_vector.pickle'

print("Loading word vector")
# wv = readWordEmb(token_dict, word2vec_emb_file, embeSize)
# pickle.dump(wv, open(word_vector_pickle, 'wb'))

wv = pickle.load(open(word_vector_pickle, 'rb'))


# Mapping Train
token_train_pos = mapWordToId(train_pos_token_list,token_dict)
dist1_train_pos = mapWordToId(train_pos_dist1_list,dist1_dict)
dist2_train_pos = mapWordToId(train_pos_dist2_list,dist2_dict)
type_train_pos = mapWordToId(train_pos_type_list, type_dict)
label_t_pos = mapLabelToId(train_pos_label_lists,label_dict)
# transform label_train into one hot
label_train_pos = np.zeros((len(label_t_pos),len(label_dict)))
for i in range(len(label_t_pos)):
    label_train_pos[i][label_t_pos[i]] = 1.0

token_train_neg1 = mapWordToId(train_neg1_token_list,token_dict)
dist1_train_neg1 = mapWordToId(train_neg1_dist1_list,dist1_dict)
dist2_train_neg1 = mapWordToId(train_neg1_dist2_list,dist2_dict)
type_train_neg1 = mapWordToId(train_neg1_type_list, type_dict)
label_t_neg1 = mapLabelToId(train_neg1_label_lists,label_dict)
# transform label_train into one hot
label_train_neg1 = np.zeros((len(label_t_neg1),len(label_dict)))
for i in range(len(label_t_neg1)):
    label_train_neg1[i][label_t_neg1[i]] = 1.0

token_train_neg2 = mapWordToId(train_neg2_token_list,token_dict)
dist1_train_neg2 = mapWordToId(train_neg2_dist1_list,dist1_dict)
dist2_train_neg2 = mapWordToId(train_neg2_dist2_list,dist2_dict)
type_train_neg2 = mapWordToId(train_neg2_type_list, type_dict)
label_t_neg2 = mapLabelToId(train_neg2_label_lists,label_dict)
# transform label_train into one hot
label_train_neg2 = np.zeros((len(label_t_neg2),len(label_dict)))
for i in range(len(label_t_neg2)):
    label_train_neg2[i][label_t_neg2[i]] = 1.0


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
token_train_pos, dist1_train_pos, dist2_train_pos, type_train_pos = \
    paddData([token_train_pos, dist1_train_pos, dist2_train_pos, type_train_pos],sentMax)
token_train_neg1, dist1_train_neg1, dist2_train_neg1, type_train_neg1 = \
    paddData([token_train_neg1, dist1_train_neg1, dist2_train_neg1, type_train_neg1],sentMax)
token_train_neg2, dist1_train_neg2, dist2_train_neg2, type_train_neg2 = \
    paddData([token_train_neg2, dist1_train_neg2, dist2_train_neg2, type_train_neg2],sentMax)
token_test, dist1_test, dist2_test, type_test = paddData([token_test, dist1_test, dist2_test, type_test],sentMax)



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
fsent = open(sent_out, 'w') # keep sentence and its results_triplet

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

train_len = len(token_train_pos)

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
    for i in range(num_batch):
        samples.append(range(batch_size*i, batch_size*(i+1)))
    samples.append(range(batch_size*(i+1), n))

    prediction = []
    for i in samples:
        p = rnn.test_step(W[i], sent_lengths[i], d1[i], d2[i], type[i], label[i])
        prediction.extend(p)
    return prediction

# training
num_batches_per_epoch = int(train_len/batch_size) + 1
iii = 0  # checkpoint number
for epoch in range(num_epochs):
    shuffle_indices = np.random.permutation(np.arange(train_len))
    token_tr_pos = token_train_pos[shuffle_indices]
    dist1_tr_pos = dist1_train_pos[shuffle_indices]
    dist2_tr_pos = dist2_train_pos[shuffle_indices]
    type_tr_pos = type_train_pos[shuffle_indices]
    label_tr_pos = label_train_pos[shuffle_indices]
    sent_tr_pos = train_pos_sent_lengths[shuffle_indices]

    token_tr_neg1 = token_train_neg1[shuffle_indices]
    dist1_tr_neg1 = dist1_train_neg1[shuffle_indices]
    dist2_tr_neg1 = dist2_train_neg1[shuffle_indices]
    type_tr_neg1 = type_train_neg1[shuffle_indices]
    label_tr_neg1 = label_train_neg1[shuffle_indices]
    sent_tr_neg1 = train_neg1_sent_lengths[shuffle_indices]

    token_tr_neg2 = token_train_neg2[shuffle_indices]
    dist1_tr_neg2 = dist1_train_neg2[shuffle_indices]
    dist2_tr_neg2 = dist2_train_neg2[shuffle_indices]
    type_tr_neg2 = type_train_neg2[shuffle_indices]
    label_tr_neg2 = label_train_neg2[shuffle_indices]
    sent_tr_neg2 = train_neg2_sent_lengths[shuffle_indices]
    loss_epoch = 0.0

    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num+1)*batch_size, train_len)
        loss = rnn.train_step(token_tr_pos[start_index:end_index], sent_tr_pos[start_index:end_index], dist1_tr_pos[start_index:end_index],
                              dist2_tr_pos[start_index:end_index], label_tr_pos[start_index:end_index],
                              token_tr_neg1[start_index:end_index], sent_tr_neg1[start_index:end_index],dist1_tr_neg1[start_index:end_index],
                              dist2_tr_neg1[start_index:end_index], label_tr_neg1[start_index:end_index],
                              token_tr_neg2[start_index:end_index], sent_tr_neg2[start_index:end_index],dist1_tr_neg2[start_index:end_index],
                              dist2_tr_neg2[start_index:end_index], label_tr_neg2[start_index:end_index],
                              drop_out)
        # print('len of loss of batch', batch_num, ': ', len(loss))
        loss_epoch += loss

    print("Epoch_:", epoch, " loss:,", loss_epoch)
    loss_list.append(round(loss_epoch, 5))

    if epoch in check_point:
    # if epoch == 1:
        iii += 1

        saver = tf.train.Saver()
        path = saver.save(rnn.sess, 'saved_models/model_'+str(iii)+'.ckpt')

        # Test
        y_pred_test = test_step(token_test, test_sent_lengths, dist1_test, dist2_test, type_test, label_test)
        y_true_test = np.argmax(label_test, 1)
        # print("y_pred_test_shape:",y_pred_test.get_shape())
        # print("y_true_test_shape:", y_true_test.get_shape())
        fscore_test.append(f1_score(y_true_test, y_pred_test, [1], average='micro'))
        test_res.append([y_true_test, y_pred_test])


index = np.argmax(fscore_test)    # Best epoch from validation set
y_true, y_pred = test_res[index]   # actual prediction


# fp.write('\n results_triplet in Test Set (Best Index) '+str(ind)+'\n')
# fp.write(str(precision_score(y_true, y_pred,[1,2,3,4], average='micro' )))
# fp.write('\t')
# fp.write(str(recall_score(y_true, y_pred, [1,2,3,4], average='micro' )))
# fp.write('\t')
# fp.write(str(f1_score(y_true, y_pred, [1,2,3,4], average='micro' )))
# fp.write('\t')
# fp.write('\n')

fp.write('\n results_triplet in Test Set (Best Index )' + str(index)+'\n')
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










