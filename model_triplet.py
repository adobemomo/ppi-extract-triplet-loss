# import tensorflow as tf
import numpy as np
import logger
import time
import sys
import os
sys.stdout = logger.Logger('logs/'+time.strftime('%Y-%m-%d %H.%M.%S',time.localtime(time.time()))+os.path.basename(__file__)+'.output', sys.stdout)
sys.stderr = logger.Logger('logs/'+time.strftime('%Y-%m-%d %H.%M.%S',time.localtime(time.time()))+os.path.basename(__file__)+'.error', sys.stderr)		# redirect std err, if necessary

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class RNN_Relation(object):
    def __init__(self, num_classes, word_dict_size, d1_dict_size, d2_dict_size, type_dict_size, sentMax, wv,
                 w_emb_size=50, d1_emb_size=5, d2_emb_size=5, type_emb_size=5, num_filters=100, l2_reg_lambda=0.0,
                 pooling='notmax'):
        tf.reset_default_graph()
        self.learning_rate_base = 0.001
        self.learning_rate_decay = 0.90
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.learning_rate_base,
                                                        self.global_step,
                                                        600,
                                                        self.learning_rate_decay
                                                        )

        self.triplet_margin = 2.5
        self.num_classes = num_classes
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.pos_sent_len = tf.placeholder(tf.int64, [None], name='pos_sent_len')
        self.pos_w = tf.placeholder(tf.int32, [None, None], name="pos_x")
        self.pos_d1 = tf.placeholder(tf.int32, [None, None], name="pos_x3")
        self.pos_d2 = tf.placeholder(tf.int32, [None, None], name='pos_x4')
        self.pos_input_y = tf.placeholder(tf.float32, [None, num_classes], name="pos_input_y")

        self.neg1_sent_len = tf.placeholder(tf.int64, [None], name='neg1_sent_len')
        self.neg1_w = tf.placeholder(tf.int32, [None, None], name="neg1_x")
        self.neg1_d1 = tf.placeholder(tf.int32, [None, None], name="neg1_x3")
        self.neg1_d2 = tf.placeholder(tf.int32, [None, None], name='neg1_x4')
        self.neg1_input_y = tf.placeholder(tf.float32, [None, num_classes], name="neg1_input_y")

        self.neg2_sent_len = tf.placeholder(tf.int64, [None], name='neg2_sent_len')
        self.neg2_w = tf.placeholder(tf.int32, [None, None], name="neg2_x")
        self.neg2_d1 = tf.placeholder(tf.int32, [None, None], name="neg2_x3")
        self.neg2_d2 = tf.placeholder(tf.int32, [None, None], name='neg2_x4')
        self.neg2_input_y = tf.placeholder(tf.float32, [None, num_classes], name="neg2_input_y")


        # Initialization
        W_wemb = tf.Variable(wv)
        W_d1emb = tf.Variable(tf.random_uniform([d1_dict_size, d1_emb_size], -1.0, +1.0))
        W_d2emb = tf.Variable(tf.random_uniform([d2_dict_size, d2_emb_size], -1.0, +1.0))
        # Embedding Layer
        pos_emb0 = tf.nn.embedding_lookup(W_wemb, self.pos_w)  # word embedding NxMx50
        pos_emb3 = tf.nn.embedding_lookup(W_d1emb, self.pos_d1)  # POS embedding  NxMx5
        pos_emb4 = tf.nn.embedding_lookup(W_d2emb, self.pos_d2)  # POS embedding  NxMx5
        pos_X = tf.concat([pos_emb0, pos_emb3, pos_emb4], 2)
        print('pos_X_shape:', pos_X.get_shape())

        neg1_emb0 = tf.nn.embedding_lookup(W_wemb, self.neg1_w)  # word embedding NxMx50
        neg1_emb3 = tf.nn.embedding_lookup(W_d1emb, self.neg1_d1)  # POS embedding  NxMx5
        neg1_emb4 = tf.nn.embedding_lookup(W_d2emb, self.neg1_d2)  # POS embedding  NxMx5
        neg1_X = tf.concat([neg1_emb0, neg1_emb3, neg1_emb4], 2)
        print('neg1_X_shape:', neg1_X.get_shape())

        neg2_emb0 = tf.nn.embedding_lookup(W_wemb, self.neg2_w)  # word embedding NxMx50
        neg2_emb3 = tf.nn.embedding_lookup(W_d1emb, self.neg2_d1)  # POS embedding  NxMx5
        neg2_emb4 = tf.nn.embedding_lookup(W_d2emb, self.neg2_d2)  # POS embedding  NxMx5
        neg2_X = tf.concat([neg2_emb0, neg2_emb3, neg2_emb4], 2)
        print('neg2_X_shape:', neg2_X.get_shape())

        self.pos_trip_emb = self.rnn(pos_X, 100, pooling, sentMax,self.pos_sent_len)
        neg1_trip_emb = self.rnn(neg1_X, 100, pooling, sentMax, self.neg1_sent_len)
        neg2_trip_emb = self.rnn(neg2_X, 100, pooling, sentMax, self.neg2_sent_len)
        print('triple_emb_shape',  self.pos_trip_emb.get_shape())
        # tripleted loss
        dis1 = tf.reduce_sum(tf.square( self.pos_trip_emb - neg1_trip_emb), 1)  # 正例与同源负例的距离
        dis2 = tf.reduce_sum(tf.square(neg1_trip_emb - neg2_trip_emb), 1)  # 同源负例与非同源负例的距离
        self.triplet_loss = tf.maximum(0.0, (self.triplet_margin + dis2 - dis1))
        # self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.triplet_loss,
        #                                                                  global_step=self.global_step)

        # classification loss
        pos_score = self.classification(self.pos_trip_emb)
        neg1_score = self.classification(neg1_trip_emb)
        neg2_score = self.classification(neg2_trip_emb)
        self.pos_prediction = tf.argmax(pos_score, 1, name="pos_predictions")
        # neg1_prediction = tf.argmax(neg1_score, 1, name="neg1_prediction")
        # neg2_prediction = tf.argmax(neg2_score, 1, name="neg2_prediction")
        pos_loss = tf.nn.softmax_cross_entropy_with_logits(logits=pos_score, labels=self.pos_input_y)
        neg1_loss = tf.nn.softmax_cross_entropy_with_logits(logits=neg1_score, labels=self.neg1_input_y)
        neg2_loss = tf.nn.softmax_cross_entropy_with_logits(logits=neg2_score, labels=self.neg2_input_y)

        self.losses = self.triplet_loss + pos_loss + neg1_loss + neg2_loss +\
                      l2_reg_lambda * (tf.nn.l2_loss(self.W) + tf.nn.l2_loss(self.b))

        ####################################
        self.losses = tf.reduce_sum(self.losses)



        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.grads_and_vars = self.optimizer.compute_gradients(self.losses)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)


        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
        session_conf = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session(config=session_conf)
        self.sess.run(tf.initialize_all_variables())


    def rnn(self, X, num_filters, pooling, sentMax, sent_len):
        # Recurrent Layer
        with tf.variable_scope('bi-lstm',reuse=tf.AUTO_REUSE):
            cell_f = tf.nn.rnn_cell.LSTMCell(num_units=100, state_is_tuple=True)
            cell_b = tf.nn.rnn_cell.LSTMCell(num_units=100, state_is_tuple=True)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_f,
                cell_bw=cell_b,
                dtype=tf.float32,
                sequence_length=sent_len,
                inputs=X
            )

            output_fw, output_bw = outputs  # NxMx100
            states_fw, states_bw = states
            print('output_fw', output_fw.get_shape())

            h = tf.concat([output_fw, output_bw], 2)  # NxMx200
            print('h', h.get_shape())

            # Attention Layer
            h = tf.expand_dims(h, -1)  # NxMx200x1
            print('h', h.get_shape())

            m = tf.reduce_max(sent_len)
            if pooling == 'max':
                pooled = tf.nn.max_pool(h, ksize=[1, sentMax, 1, 1], strides=[1, 1, 1, 1], padding='VALID',
                                        name="pool")  # Nx1x200x1
            else:
                pooled = tf.reduce_sum(h, 1)
            # pooled = tf.nn.avg_pool(h, ksize=[1, sentMax, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")	#Nx1x200x1
            print('pooled', pooled.get_shape())

            h2 = tf.reshape(pooled, [-1, 2 * num_filters])  # ?x200
            print('h2', h2.get_shape())

            # dropout layer
            h2 = tf.nn.dropout(h2, self.dropout_keep_prob)
            h2 = tf.tanh(h2)

            self.W = tf.Variable(tf.truncated_normal([2 * num_filters, 25], stddev=0.1), name="W")
            self.b = tf.Variable(tf.constant(0.1, shape=[25]), name="b")

            emb = tf.nn.xw_plus_b(h2, self.W, self.b, name="emb")

        return emb

    def classification(self, triplet_emb):
        w1 = tf.Variable(tf.truncated_normal([25, 10], stddev=0.1), name="w1")
        b1 = tf.Variable(tf.constant(0.1, shape=[10]), name="b1")
        w2 = tf.Variable(tf.truncated_normal([10, self.num_classes], stddev=0.1), name="w2")
        b2 = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b2")
        output1 = tf.nn.xw_plus_b(triplet_emb, w1, b1, name="output1")
        output1 = tf.nn.relu(output1)
        output2 = tf.nn.xw_plus_b(output1, w2, b2, name="output2")
        return output2

    def train_step(self, pos_W_batch, pos_Sent_len, pos_d1_batch, pos_d2_batch, pos_y_batch,
                   neg1_W_batch, neg1_Sent_len, neg1_d1_batch, neg1_d2_batch, neg1_y_batch,
                   neg2_W_batch, neg2_Sent_len, neg2_d1_batch, neg2_d2_batch, neg2_y_batch,
                   drop_out):
        # Padding data
        feed_dict = {
            self.pos_w: pos_W_batch,
            self.pos_d1: pos_d1_batch,
            self.pos_d2: pos_d2_batch,
            self.pos_sent_len: pos_Sent_len,
            self.pos_input_y: pos_y_batch,
            self.neg1_w: neg1_W_batch,
            self.neg1_d1: neg1_d1_batch,
            self.neg1_d2: neg1_d2_batch,
            self.neg1_sent_len: neg1_Sent_len,
            self.neg1_input_y: neg1_y_batch,
            self.neg2_w: neg2_W_batch,
            self.neg2_d1: neg2_d1_batch,
            self.neg2_d2: neg2_d2_batch,
            self.neg2_sent_len: neg2_Sent_len,
            self.neg2_input_y: neg2_y_batch,
            self.dropout_keep_prob: drop_out
        }
        _, step, loss = \
            self.sess.run([self.train_op, self.global_step, self.losses], feed_dict)
        return loss

    def test_step(self, W_batch, Sent_len, d1_batch, d2_batch, t_batch, y_batch):

        #		w,d1,d2,typet = paddData([W_batch, d1_batch, d2_batch, t_batch])
        feed_dict = {
            self.pos_w: W_batch,
            self.pos_d1: d1_batch,
            self.pos_d2: d2_batch,
            #self.type	:t_batch,
            self.pos_sent_len: Sent_len,
            self.dropout_keep_prob: 1.0,
            self.pos_input_y: y_batch
        }
        step, loss, predictions = \
            self.sess.run([self.global_step, self.losses, self.pos_prediction], feed_dict)

        # print "Accuracy in test data", accuracy
        return predictions