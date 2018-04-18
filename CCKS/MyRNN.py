import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper
from utils import Utils
from utils import batch_generator,predict_generator
import numpy as np
import os

U = Utils()


class MyMode:
    def __init__(self,n_steps,n_hiddens,n_class,input_embedding_size,LR,vocab_size,other_vocab_size
                 ,other_embedding_size,use_crf=False,isTrain = True):
        # self.n_steps = n_steps
        self.n_hiddens = n_hiddens
        self.n_class = n_class
        self.input_embedding_size = input_embedding_size
        self.learning = LR
        self.vocab_size = vocab_size
        self.other_vocab_size = other_vocab_size
        self.other_embedding_size = other_embedding_size
        self.use_crf = use_crf
        self.isTrain = isTrain

        self.buildInput()
        self.embedding()
        self.buildlstm()
        self.build_pre_op()
        self.buildLoss()
        self.build_optimizer()
        self.saver = tf.train.Saver()





    def buildInput(self):
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(dtype=tf.int32,shape=
                (None,None),name='input')

            self.others = tf.placeholder(dtype=tf.int32,shape=
            (None,None),name = 'other')
            self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                                                   name="sequence_lengths")
            self.targets = tf.placeholder(dtype=tf.int32,shape=
                (None,None),name='targets')

            self.keep_prob = tf.placeholder(dtype=tf.float32,name='keep_prob')


    # def getfeed(self,X,M,Y):
    #     X,length = U.paddingList(X,pad_token=0)
    #     feed = {
    #
    #
    #     }





    def embedding(self):
            _input_embedding = tf.get_variable('input_embedding',dtype=tf.float32,shape = (self.vocab_size
                                            ,self.input_embedding_size))
            inputs_embedding = tf.nn.embedding_lookup(_input_embedding,self.inputs)

            _other_embedding = tf.get_variable('other_embedding',dtype=tf.float32,shape=(47
            ,self.other_embedding_size))

            other_embedding = tf.nn.embedding_lookup(_other_embedding,self.others)

            if self.isTrain:
                self.lstm_inputs = tf.concat([inputs_embedding,other_embedding],axis=-1)

            # self.lstm_inputs = inputs_embedding





    def buildlstm(self):
        cell_forward = tf.contrib.rnn.BasicLSTMCell(self.n_hiddens)
        cell_backward = tf.contrib.rnn.BasicLSTMCell(self.n_hiddens)

        orginal_lstm_outputs,nextstate = tf.nn.bidirectional_dynamic_rnn(cell_forward,cell_backward
                                ,self.lstm_inputs,dtype=tf.float32)
        forward_out, backward_out  = orginal_lstm_outputs




        self.lstm_outputs = tf.concat([forward_out,backward_out],axis=2)


        outpus2D = tf.reshape(self.lstm_outputs,shape=(-1,2*self.n_hiddens))

        self.outs = tf.layers.dense(outpus2D,self.n_class)
        # print('outs',self.outs)

        self.logits = tf.reshape(self.outs,shape=(-1,self.n_steps,self.n_class))
        # print('11', self.logits)
        # self.logits = tf.reshape(self.outs,[-1,self.n_steps,self.n_class])
        # print('22', self.logits)

    def build_pre_op(self):
        if self.use_crf == False:
            self.labels_pre = tf.cast(tf.argmax(self.logits,axis=-1),tf.int32)



    def buildLoss(self):
        # print('33', self.targets)
        #
        if(self.use_crf == False):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,labels=self.targets)
            # print('44',self.logits,self.targets)
            self.loss = tf.reduce_mean(loss)
        else:
            pass

        # tf.contrib.crf.crf _


    def build_optimizer(self):
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.05).minimize(self.loss)

        # tf.train.
        pass

    def train(self,batch_generator):

        sess = self.session
        sess.run(tf.global_variables_initializer())
        step = 0
        losses = []
        for X, Y, M in batch_generator:
            # print('Y', Y)
            step += 1
            X,sequence_lengths = U.paddingList(X,0)


            feed = {self.inputs: X,
                    self.others: M,
                    self.targets: Y
                    }
            batch_loss, logits, _ = sess.run([self.loss, self.logits, self.train_op], feed_dict=feed)

            if (step % 1 == 0):
                print('loss', batch_loss)
                losses.append(batch_loss)

            if (step % 10 == 0):
                # print('asdasdasd')
                # self.saver.save(sess, os.path.join('./model/'), global_step=step)
                # file = open('./resource/losses.txt', 'w', encoding='utf-8')
                # file.write(str(losses))
                # file.close()
                return
            if (step == 10000):
                return


    def predict(self,predict_generator):
        sess = self.session
        sess.run(tf.global_variables_initializer())
        for X, M in predict_generator:
            feed = {self.inputs: X,
                    self.others: M}

            labels_pre = sess.run([self.labels_pre], feed_dict=feed)
            # print(labels_pre)
            # print(type(labels_pre))
            # print(type(np.array(labels_pre)))
            # print(np.array(labels_pre).shape)
            # print(type(labels_pre))

            # print(list(labels_pre[0][0]))
            preds = [U.int_to_target(idx) for idx in list(labels_pre[0][0])]
            print(preds)
            # # print(labels_pre.shape)
            # predict_generator(preds)

            # labels_pre = tf.reshape(labels_pre,shape=(20,310))

            pass


    def load(self, checkpoint):
        self.session = tf.Session()
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))



if __name__ == '__main__':
    pass
    g = batch_generator(500,'all.txt')
    rnn = MyMode(310,128,20,128,0.005,1723,47,128)
    rnn.load(tf.train.latest_checkpoint('./model/'))

    rnn.train(g)
    # rnn.load(tf.train.latest_checkpoint('./model/'))
    pre = predict_generator()
    rnn.predict(pre)
    pass

