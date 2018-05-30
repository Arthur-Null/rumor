import time
from datetime import timedelta

import tensorflow as tf
import numpy as np
import os
import pickle as pkl
from sklearn.metrics import *


def getbatch(batchsize, f):
    data = []
    seqlen = []
    label = []
    for i in range(batchsize):
        try:
            x = pkl.load(f).tolist()
            y = pkl.load(f)
        except:
            break
        if len(x) < 25:
            seqlen.append(len(x))
            for _ in range(len(x), 25):
                x.append([0.] * 5000)
        else:
            seqlen.append(25)
            x = x[:25]
        data.append(x)
        # if (y == 1):
        #     label.append([0, 1])
        # else:
        #     label.append([1, 0])
        label.append(y)
    return data, label, seqlen




class rnn:
    def __init__(self, path, trainset, testset, para):
        self.graph = tf.Graph()

        self._path = path
        self.trainset = trainset
        self.testset = testset
        self.para = para
        self._save_path, self._logs_path = None, None
        self.loss, self.train_step, self.prediction = None, None, None
        with self.graph.as_default():
            self._define_inputs()
            self._build_graph()
            self.initializer = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
        self._initialize_session()


    @property
    def save_path(self):
        if self._save_path is None:
            save_path = '%s/checkpoint' % self._path
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path = os.path.join(save_path, 'model.ckpt')
            self._save_path = save_path
        return self._save_path

    @property
    def logs_path(self):
        if self._logs_path is None:
            logs_path = '%s/logs' % self._path
            if not os.path.exists(logs_path):
                os.makedirs(logs_path)
            self._logs_path = logs_path
        return self._logs_path

    def save_model(self, global_step=None):
        self.saver.save(self.sess, self.save_path, global_step=global_step)

    def _define_inputs(self):
        self.input = tf.placeholder(
            tf.float32,
            shape=[None, 25, 5000]
        )
        self.labels = tf.placeholder(
            tf.float32,
            shape=[None]
        )
        self.seqlen = tf.placeholder(
            tf.int32,
            shape=[None],
            name='seqlen'
        )
        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')

    def _initialize_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)

        self.sess.run(self.initializer)

    def _build_graph(self):
        x = self.input
        x = tf.layers.dense(x, self.para['embedding_size'], tf.nn.relu)
        gru_cell = tf.contrib.rnn.GRUCell(self.para['hidden_size'])
        gru_cell = tf.nn.rnn_cell.DropoutWrapper(gru_cell, input_keep_prob=self.keep_prob,
                                                 output_keep_prob=self.keep_prob)
        states_h, last_h = tf.nn.dynamic_rnn(gru_cell, x, self.seqlen, dtype=tf.float32)
        output = tf.layers.dense(last_h, 2, tf.nn.softmax)
        pred = output[:, 1]
        self.prediction = output
        loss = tf.losses.log_loss(self.labels, pred)
        self.loss = loss
        optimizer = tf.train.AdamOptimizer(learning_rate=self.para['lr'])
        self.train_step = optimizer.minimize(loss)

    def train_one_epoch(self):
        fin = open(self.trainset, 'rb')
        losses = []
        preds = []
        labels = []
        while True:
            batch = getbatch(self.para['batch_size'], fin)
            data, label, seqlen = batch
            if len(label) != self.para['batch_size']:
                break
            feed_dict = {
                self.input: data,
                self.labels: label,
                self.seqlen: seqlen,
                self.is_training: True,
                self.keep_prob: 0.5
            }
            fetches = [self.train_step, self.loss, self.prediction]
            result = self.sess.run(fetches, feed_dict=feed_dict)
            _, loss, pred = result
            losses.append(loss)
            preds += pred.tolist()
            labels += label
        self.save_model()
        preds = np.array(preds)
        auc = roc_auc_score(labels, preds[:, 1])
        acc = accuracy_score(labels, np.argmax(preds, axis=1))
        print("AUC = " + "{:.4f}".format(auc))
        print("Accuracy = " + "{:.4f}".format(acc))
        print("Loss = " + "{:.4f}".format(np.mean(losses)))
        return np.mean(losses), auc, acc

    def test(self):
        fin = open(self.testset, 'rb')
        losses = []
        preds = []
        labels = []
        while True:
            batch = getbatch(self.para['batch_size'], fin)
            data, label, seqlen = batch
            if len(label) != self.para['batch_size']:
                break
            feed_dict = {
                self.input: data,
                self.labels: label,
                self.seqlen: seqlen,
                self.is_training: True,
                self.keep_prob: 0.5
            }
            fetches = [self.loss, self.prediction]
            result = self.sess.run(fetches, feed_dict=feed_dict)
            loss, pred = result
            losses.append(loss)
            preds += pred.tolist()
            labels += label
        preds = np.array(preds)
        auc = roc_auc_score(labels, preds[:, 1])
        acc = accuracy_score(labels, np.argmax(preds, axis=1))
        print("AUC = " + "{:.4f}".format(auc))
        print("Accuracy = " + "{:.4f}".format(acc))
        print("Loss = " + "{:.4f}".format(np.mean(losses)))
        return np.mean(losses), auc, acc

    def log(self, epoch, result, prefix):
        s = prefix + '\t' + str(epoch)
        for i in result:
            s += ('\t' + str(i))
        fout = open(self.logs_path + 'log','a')
        fout.write(s + '\n')

    def train_until_cov(self):
        epoch = 0
        losses = []
        total_start_time = time.time()
        while True:
            epoch += 1
            print('-' * 30, 'Train epoch: %d' % epoch, '-' * 30)
            start_time = time.time()
            result = self.train_one_epoch()
            self.log(epoch, result, 'train')
            print('Time per train epoch: %s' % (
                str(timedelta(seconds=time.time()-start_time))
            ))
            start_time = time.time()
            result = self.test()
            losses.append(result[0])
            self.log(epoch, result, 'test')
            print('Time per test epoch: %s' % (
                str(timedelta(seconds=time.time() - start_time))
            ))
            if epoch > 5 and losses[-1] > losses[-2] > losses[-3]:
                break

        total_training_time = time.time() - total_start_time
        print('\nTotal training time: %s' % str(timedelta(seconds=total_training_time)))

    def load_model(self):
        try:
            self.saver.restore(self.sess, self.save_path)
        except Exception:
            raise IOError('Failed to load model from save path: %s' % self.save_path)
        print('Successfully load model from save path: %s' % self.save_path)


if __name__ == '__main__':
    para = {'batch_size': 20, 'lr': 0.01, 'hidden_size': 500, 'embedding_size': 100}
    path = './model'
    trainset = 'dataset/train.pkl'
    testset = 'dataset/test.pkl'
    model = rnn(path, trainset, testset, para)
    model.train_until_cov()