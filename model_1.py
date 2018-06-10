# encoding=utf-8
# Project: transfer_cws
# Author: xingjunjie
# Create Time: 30/11/2017 9:15 AM on PyCharm

import tensorflow as tf
from utils import Progbar
from data_utils import pad_sequences, minibatches, get_chunks, minibatches_evaluate
import numpy as np
import os
from functools import partial
from penalty import MKL, CMD, MMD, gaussian_kernel_matrix, _de_pad


class Model(object):
    def __init__(self, args, ntags, nwords, ntarwords=None, src_embedding=None,
                 target_embedding=None, logger=None, src_batch_size=None):
        self.args = args
        self.src_embedding = src_embedding
        self.target_embedding = target_embedding
        self.ntags = ntags
        self.nwords = nwords
        self.ntarwords = ntarwords
        self.logger = logger
        self.init_lr = args.learning_rate
        self.src_batch_size = src_batch_size
        self.target_batch_size = self.args.batch_size - self.src_batch_size

        self.describe = "parallel training, only with mmd, model-1"

        self.initializer = tf.contrib.layers.xavier_initializer()
        self.l2_regularizer = tf.contrib.layers.l2_regularizer(self.args.l2_ratio)

        self.info = {
            'dev': [],
            'train': [],
            'loss': [],
            'test': None
        }

    def add_placeholder(self):
        with tf.device('/gpu:{:d}'.format(self.args.gpu_device)):
            self.batch_size = tf.placeholder(tf.int32, shape=[])

            # shape = [batch size, max length of sequence in batch]
            self.src_word_ids = tf.placeholder(tf.int32, shape=[None, None])

            # shape = [batch size, max length of sequence in batch]
            self.target_word_ids = tf.placeholder(tf.int32, shape=[None, None])

            # shape = [batch size]
            self.sequence_lengths = tf.placeholder(tf.int32, shape=[None])

            # shape = [batch size]
            self.src_sequence_lengths = tf.placeholder(tf.int32, shape=[None])

            # shape = [batch size]
            self.target_sequence_lengths = tf.placeholder(tf.int32, shape=[None])

            # shape = [batch size, max length of sequence in batch]
            self.labels = tf.placeholder(tf.int32, shape=[None, None])

            # hyper parameters
            self.dropout = tf.placeholder(tf.float32, shape=[])

            self.lr = tf.placeholder(tf.float32, shape=[])

            self.is_training = tf.placeholder(tf.bool)

    def get_feed_dict(self, sentences, labels, target_words, lr=None, dropout=None, src_batch_size=None, mode="all",
                      is_training=True):
        if mode == 'all':
            all_words_ids, sequence_lengths = pad_sequences(sentences + target_words, pad_tok=0)

            words_ids = all_words_ids[:src_batch_size] + [[0] * len(all_words_ids[0])] * (
                    self.args.batch_size - src_batch_size)
            src_sequence_lengths = sequence_lengths[:src_batch_size] + [0] * (self.args.batch_size - src_batch_size)
            target_words_ids = [[0] * len(all_words_ids[0])] * src_batch_size + all_words_ids[src_batch_size:]
            target_sequence_lengths = [0] * src_batch_size + sequence_lengths[src_batch_size:]

            feed_dict = {
                self.src_word_ids: words_ids,
                self.src_sequence_lengths: src_sequence_lengths,
                self.target_word_ids: target_words_ids,
                self.target_sequence_lengths: target_sequence_lengths,
                self.sequence_lengths: sequence_lengths,
                self.batch_size: self.args.batch_size,
                self.is_training: is_training,
            }
        elif mode == 'target':
            target_words_ids, target_sequence_lengths = pad_sequences(target_words, pad_tok=0)
            sequence_lengths = target_sequence_lengths
            feed_dict = {
                self.src_word_ids: np.zeros_like(target_words_ids),
                self.src_sequence_lengths: np.zeros_like(target_sequence_lengths),
                self.target_word_ids: target_words_ids,
                self.target_sequence_lengths: target_sequence_lengths,
                self.sequence_lengths: target_sequence_lengths,
                self.batch_size: self.args.batch_size,
                self.is_training: is_training,
            }

        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed_dict[self.labels] = labels

        if lr is not None:
            feed_dict[self.lr] = lr

        if dropout is not None:
            feed_dict[self.dropout] = dropout

        return feed_dict, sequence_lengths

    def add_src_word_embeddings_op(self):
        with tf.device('/gpu:{:d}'.format(self.args.gpu_device)):
            with tf.variable_scope("src_word"):
                _word_embeddings = tf.get_variable('embedding', shape=[self.nwords, self.args.embedding_size],
                                                   initializer=self.initializer,
                                                   trainable=not self.args.disable_src_embed_training,
                                                   regularizer=self.l2_regularizer)
                word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.src_word_ids)

                if self.args.share_embed:
                    target_word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.target_word_ids)
                    self.target_word_embeddings = tf.nn.dropout(target_word_embeddings, self.dropout)

            if self.args.use_pretrain_src:
                pre_train_size = self.src_embedding.shape[0]
                self.src_embedding_init = _word_embeddings[:pre_train_size].assign(self.src_embedding)

            self.src_word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)

    def add_target_word_embeddings_op(self):
        with tf.device('/gpu:{:d}'.format(self.args.gpu_device)):
            with tf.variable_scope("target_word"):
                _word_embeddings = tf.get_variable('embedding', shape=[self.ntarwords, self.args.embedding_size],
                                                   initializer=self.initializer, regularizer=self.l2_regularizer)
                word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.target_word_ids)

            if self.args.use_pretrain_target:
                pre_train_size = self.target_embedding.shape[0]
                self.target_embedding_init = _word_embeddings[:pre_train_size].assign(self.target_embedding)

            self.target_word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)

    def add_logits_op(self):
        with tf.device('/gpu:{:d}'.format(self.args.gpu_device)):
            with tf.variable_scope('src_lstm'):
                cell_fw = tf.contrib.rnn.LSTMCell(self.args.lstm_hidden)
                cell_bw = tf.contrib.rnn.LSTMCell(self.args.lstm_hidden)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.src_word_embeddings, sequence_length=self.src_sequence_lengths,
                    dtype=tf.float32)
                outout = tf.concat([output_fw, output_bw], axis=-1)
                outout = tf.nn.dropout(outout, self.dropout)

                self.src_after_specific = outout

            with tf.variable_scope('target_lstm'):
                cell_fw = tf.contrib.rnn.LSTMCell(self.args.lstm_hidden)
                cell_bw = tf.contrib.rnn.LSTMCell(self.args.lstm_hidden)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.target_word_embeddings, sequence_length=self.target_sequence_lengths,
                    dtype=tf.float32)
                outout = tf.concat([output_fw, output_bw], axis=-1)
                outout = tf.nn.dropout(outout, self.dropout)

                self.target_after_specific = outout

            with tf.variable_scope('src_lstm_linear'):
                W = tf.get_variable("W", shape=[2 * self.args.lstm_hidden, self.ntags],
                                    dtype=tf.float32, initializer=self.initializer, regularizer=self.l2_regularizer)
                b = tf.get_variable("b", shape=[self.ntags], dtype=tf.float32,
                                    initializer=self.initializer, regularizer=self.l2_regularizer)
                ntime_steps = tf.shape(self.src_after_specific)[1]
                output = tf.reshape(self.src_after_specific, [-1, 2 * self.args.lstm_hidden])
                pred = tf.matmul(output, W) + b
                self.src_logits = tf.reshape(pred, [-1, ntime_steps, self.ntags])

            with tf.variable_scope('target_lstm_linear'):
                W = tf.get_variable("W", shape=[2 * self.args.lstm_hidden, self.ntags],
                                    dtype=tf.float32, initializer=self.initializer, regularizer=self.l2_regularizer)
                b = tf.get_variable("b", shape=[self.ntags], dtype=tf.float32,
                                    initializer=self.initializer, regularizer=self.l2_regularizer)
                ntime_steps = tf.shape(self.target_after_specific)[1]
                output = tf.reshape(self.target_after_specific, [-1, 2 * self.args.lstm_hidden])
                pred = tf.matmul(output, W) + b
                self.target_logits = tf.reshape(pred, [-1, ntime_steps, self.ntags])

    def add_loss_op(self):
        with tf.device('/gpu:{:d}'.format(self.args.gpu_device)):
            # CRF loss
            with tf.variable_scope('src_crf'):
                self.src_log_likelihood, self.src_transition_params = tf.contrib.crf.crf_log_likelihood(
                    self.src_logits, self.labels, self.src_sequence_lengths
                )

            with tf.variable_scope('target_crf'):
                self.target_log_likelihood, self.target_transition_params = tf.contrib.crf.crf_log_likelihood(
                    self.target_logits, self.labels, self.target_sequence_lengths
                )

            self.src_crf_loss = tf.reduce_mean(-self.src_log_likelihood[:self.src_batch_size])
            self.target_crf_loss = tf.reduce_mean(-self.target_log_likelihood[self.src_batch_size:])

            # MMD loss
            if self.args.penalty_ratio > 0:
                self.src_depad = _de_pad(self.src_after_specific, self.src_sequence_lengths)
                self.target_depad = _de_pad(self.target_after_specific, self.target_sequence_lengths)

                if self.args.penalty == 'mmd':
                    with tf.name_scope('mmd'):
                        sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5,
                                  1e6]
                        gaussian_kernel = partial(gaussian_kernel_matrix, sigmas=tf.constant(sigmas))
                        loss_value = MMD(self.src_depad, self.target_depad, kernel=gaussian_kernel)
                        mmd_loss = tf.maximum(1e-4, loss_value)

                    self.penalty_loss = self.args.penalty_ratio * mmd_loss
                elif self.args.penalty == 'kl':
                    self.src_depad_sm = tf.nn.softmax(self.src_depad)
                    self.target_depad_sm = tf.nn.softmax(self.target_depad)
                    self.kl_loss = MKL(self.src_depad_sm, self.target_depad_sm)
                    self.penalty_loss = self.args.penalty_ratio * self.kl_loss
                elif self.args.penalty == 'cmd':
                    self.cmd_loss = CMD(self.src_depad, self.target_depad, 5)
                    self.penalty_loss = self.args.penalty_ratio * self.cmd_loss
                else:
                    self.logger.critical("Penalty Type Invalid.")
                    exit(9)

                temp = self.src_crf_loss + self.target_crf_loss + self.penalty_loss
            else:
                temp = self.src_crf_loss + self.target_crf_loss

            if self.args.use_l2:
                self.l2_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                temp1 = temp + self.l2_loss
            else:
                temp1 = temp

            if not self.args.share_crf:
                self.crf_l2_loss = tf.nn.l2_loss(
                    self.target_transition_params - self.src_transition_params) * self.args.crf_l2_ratio
                temp2 = temp1 + self.crf_l2_loss
            else:
                temp2 = temp1

            self.loss = temp2

    def add_train_op(self):
        with tf.device('/gpu:{:d}'.format(self.args.gpu_device)):
            with tf.variable_scope('train'):
                if self.args.optim.lower() == 'adam':
                    optimizer = tf.train.AdamOptimizer(self.lr)
                elif self.args.optim.lower() == 'sgd':
                    optimizer = tf.train.GradientDescentOptimizer(self.lr)
                else:
                    raise NotImplementedError("Unknown optim {}".format(self.args.optim))

                self.train_op = optimizer.minimize(self.loss)

    def add_init_op(self):
        self.init = tf.global_variables_initializer()

    def build(self):
        self.add_placeholder()
        self.add_src_word_embeddings_op()
        if not self.args.share_embed:
            self.add_target_word_embeddings_op()
        self.add_logits_op()
        self.add_loss_op()
        self.add_train_op()
        self.add_init_op()
        self.logger.info("Model info: {}".format(self.describe))

    def predict_batch(self, sess, words=None, target_words=None, mode='target', is_training=True):
        feed_dict, sequence_lengths = self.get_feed_dict(words, None, target_words=target_words, dropout=1.0, mode=mode,
                                                         is_training=is_training)

        viterbi_sequences = []
        logits, transition_params = sess.run([self.target_logits, self.target_transition_params],
                                             feed_dict=feed_dict)
        for logit, sequence_length in zip(logits, sequence_lengths):
            logit = logit[:sequence_length]
            viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(
                logit, transition_params
            )
            viterbi_sequences += [viterbi_sequence]

        return viterbi_sequences, sequence_lengths

    def run_epoch(self, sess, src_train, src_dev, tags, target_train, target_dev, n_epoch_noimprove):
        nbatces = (len(target_train) + self.target_batch_size - 1) // self.target_batch_size
        prog = Progbar(target=nbatces)
        total_loss = 0

        src = minibatches(src_train, self.src_batch_size, circle=True)
        target = minibatches(target_train, self.target_batch_size, circle=True)

        for i in range(nbatces):
            src_words, src_tags, _ = next(src)
            target_words, target_tags, _ = next(target)
            labels = src_tags + target_tags

            feed_dict, _ = self.get_feed_dict(src_words, labels, target_words, self.args.learning_rate,
                                              self.args.dropout, self.src_batch_size, is_training=True)

            if self.args.penalty_ratio > 0:
                _, src_crf_loss, target_crf_loss, penalty_loss, loss = sess.run(
                    [self.train_op, self.src_crf_loss, self.target_crf_loss, self.penalty_loss, self.loss],
                    feed_dict=feed_dict)
                try:
                    prog.update(i + 1,
                                [("train loss", loss[0]), ("src crf", src_crf_loss), ("target crf", target_crf_loss),
                                 ("{} loss".format(self.args.penalty), penalty_loss)])
                except:
                    prog.update(i + 1,
                                [("train loss", loss), ("src crf", src_crf_loss), ("target crf", target_crf_loss),
                                 ("{} loss".format(self.args.penalty), penalty_loss)])
            else:
                _, src_crf_loss, target_crf_loss, loss = sess.run(
                    [self.train_op, self.src_crf_loss, self.target_crf_loss, self.loss],
                    feed_dict=feed_dict)
                try:
                    prog.update(i + 1,
                                [("train loss", loss[0]), ("src crf", src_crf_loss), ("target crf", target_crf_loss)])
                except:
                    prog.update(i + 1,
                                [("train loss", loss), ("src crf", src_crf_loss), ("target crf", target_crf_loss)])
            total_loss += loss

        self.info['loss'] += [total_loss / nbatces]
        acc, p, r, f1 = self.run_evaluate(sess, target_train, tags, target='target')
        self.info['dev'].append((acc, p, r, f1))
        self.logger.critical(
            "target train acc {:04.2f}  f1  {:04.2f}  p {:04.2f}  r  {:04.2f}".format(100 * acc, 100 * f1, 100 * p,
                                                                                      100 * r))
        acc, p, r, f1 = self.run_evaluate(sess, target_dev, tags, target='target')
        self.info['dev'].append((acc, p, r, f1))
        self.logger.info(
            "dev acc {:04.2f}  f1  {:04.2f}  p {:04.2f}  r  {:04.2f}".format(100 * acc, 100 * f1, 100 * p, 100 * r))
        return acc, p, r, f1

    def run_evaluate(self, sess, test, tags, target='src'):
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        nbatces = (len(test) + self.args.batch_size - 1) // self.args.batch_size
        prog = Progbar(target=nbatces)
        for i, (words, labels, target_words) in enumerate(minibatches(test, self.args.batch_size)):
            if target == 'src':
                labels_pred, sequence_lengths = self.predict_batch(sess, words, mode=target, is_training=False)
            else:
                labels_pred, sequence_lengths = self.predict_batch(sess, None, words, mode=target, is_training=False)

            for lab, label_pred, length in zip(labels, labels_pred, sequence_lengths):
                lab = lab[:length]
                lab_pred = label_pred[:length]
                accs += [a == b for (a, b) in zip(lab, lab_pred)]
                lab_chunks = set(get_chunks(lab, tags))
                lab_pred_chunks = set(get_chunks(lab_pred, tags))
                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

            prog.update(i + 1)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)
        return acc, p, r, f1

    def predict(self, sess, test, id_to_tag, id_to_word):
        nbatces = (len(test) + self.args.batch_size - 1) // self.args.batch_size
        prog = Progbar(target=nbatces)
        with open(self.args.predict_out, 'w+', encoding='utf8') as outfile:
            for i, (words, target_words, true_words) in enumerate(minibatches_evaluate(test, self.args.batch_size)):
                labels_pred, sequence_lengths = self.predict_batch(sess, words)

                for word, true_word, label_pred, length in zip(words, true_words, labels_pred, sequence_lengths):
                    true_word = true_word[:length]
                    lab_pred = label_pred[:length]

                    for item, tag in zip(true_word, lab_pred):
                        outfile.write(item + '\t' + id_to_tag[tag] + '\n')
                    outfile.write('\n')

                prog.update(i + 1)

    def train(self, src_train, src_dev, tags, target_train, target_dev, src_batch_size, target_batch_size):
        best_score = -1e-4
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_config.gpu_options.per_process_gpu_memory_fraction = self.args.gpu_frac
        tf_config.allow_soft_placement = True
        with tf.Session(config=tf_config) as sess:
            sess.run(self.init)
            if self.args.use_pretrain_src:
                sess.run(self.src_embedding_init)
            if self.args.use_pretrain_target and self.args.flag == 1:
                sess.run(self.target_embedding_init)

            nepoch_no_imprv = 0
            for epoch in range(self.args.epoch):
                self.logger.info("Epoch : {}/{}".format(epoch + 1, self.args.epoch))

                acc, p, r, f1 = self.run_epoch(sess, src_train, src_dev, tags, target_train, target_dev,
                                               nepoch_no_imprv)

                self.args.learning_rate *= self.args.lr_decay

                if f1 > best_score:
                    nepoch_no_imprv = 0
                    if not os.path.exists(self.args.model_output):
                        os.makedirs(self.args.model_output)
                    saver = tf.train.Saver()
                    saver.save(sess, self.args.model_output)
                    best_score = f1
                    self.logger.info("New best score: {}".format(f1))
                else:
                    nepoch_no_imprv += 1
                    if nepoch_no_imprv >= self.args.nepoch_no_imprv:
                        self.logger.info("Early stopping {} epochs without improvement".format(nepoch_no_imprv))
                        break

        return self.evaluate(target_dev, tags, target='target')

    def evaluate(self, test, tags, target='src'):
        saver = tf.train.Saver()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_config.gpu_options.per_process_gpu_memory_fraction = self.args.gpu_frac
        tf_config.allow_soft_placement = True
        with tf.Session(config=tf_config) as sess:
            self.logger.info("Testing model over test set")
            saver.restore(sess, self.args.model_output)
            acc, p, r, f1 = self.run_evaluate(sess, test, tags, target=target)
            self.info['test'] = (acc, p, r, f1)
            self.logger.info("- test acc {:04.2f} - f1 {:04.2f}".format(100 * acc, 100 * f1))
        return acc, p, r, f1
