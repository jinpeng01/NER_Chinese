import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper
from utils import Utils
from utils import batch_generator,predict_generator
import numpy as np
import os

U = Utils()


class MyMode:
    def __init__(self,n_steps,n_hiddens,n_class,input_embedding_size,LR,vocab_size,other_vocab_size
                 ,other_embedding_size,use_crf=True,batch_size = 20):
        self.n_steps = n_steps
        self.n_hiddens = n_hiddens
        self.n_class = n_class
        self.input_embedding_size = input_embedding_size
        self.learning = LR
        self.vocab_size = vocab_size
        self.other_vocab_size = other_vocab_size
        self.other_embedding_size = other_embedding_size
        self.use_crf = use_crf
        self.batch_size = batch_size
        self.isTrain = 0
        self.use_Other = 0

        self.buildInput()
        self.n_steps = tf.shape(self.inputs)[-1]
        self.embedding()
        self.buildlstm()
        self.build_pre_op()
        self.buildLoss()
        self.build_optimizer()

        self.saver = tf.train.Saver()
    def buildInput(self):
        self.inputs = tf.placeholder(dtype=tf.int32, shape=
        [None, None], name='inputs')

        self.others = tf.placeholder(dtype=tf.int32, shape=
        [None, None], name='others')

        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                                               name="sequence_lengths")
        self.targets = tf.placeholder(dtype=tf.int32, shape=
        [None, None], name='targets')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')


    def getTrainfeed(self, X, M, Y, keep_prob=None):
        X_, lengths = U.paddingList(X, pad_token=0)
        M_, _ = U.paddingList(M, pad_token=0)
        Y_, _ = U.paddingList(Y, pad_token=0)
        feed = {
            self.inputs: X_,
            self.sequence_lengths: lengths,
            self.others: M_,
            self.targets: Y_,
            self.keep_prob: 0.5,
        }
        return feed, lengths

    def getPredictfeed(self, X, M):
        X, lengths = U.paddingList(X, pad_token=0)
        M, _ = U.paddingList(M, pad_token=0)
        feed = {
            self.inputs: X,
            self.sequence_lengths: lengths,
            self.others: M,
        }
        return feed, lengths

    def embedding(self):

            _input_embedding = tf.get_variable('input_embedding',dtype=tf.float32,shape = (self.vocab_size
                                            ,self.input_embedding_size))
            self.inputs_embedding = tf.nn.embedding_lookup(_input_embedding,self.inputs)

            if(self.use_Other == 1):
                _other_embedding = tf.get_variable('other_embedding',dtype=tf.float32,shape=(47
                ,self.other_embedding_size))

                self.other_embedding = tf.nn.embedding_lookup(_other_embedding,self.others)

                self.lstm_inputs = tf.concat([self.inputs_embedding,self.other_embedding],axis=2)
            else:
                self.lstm_inputs = self.inputs_embedding

            if (self.isTrain == 1):
                print('isTrain --------------------')
                self.lstm_inputs = tf.nn.dropout(self.lstm_inputs,self.keep_prob)

            # self.lstm_inputs = inputs_embedding




    def buildlstm(self):
        cell_forward = tf.contrib.rnn.BasicLSTMCell(self.n_hiddens)
        cell_backward = tf.contrib.rnn.BasicLSTMCell(self.n_hiddens)

        orginal_lstm_outputs,nextstate = tf.nn.bidirectional_dynamic_rnn(cell_forward,cell_backward
                                ,self.lstm_inputs,dtype=tf.float32)
        forward_out, backward_out  = orginal_lstm_outputs
        self.lstm_outputs = tf.concat([forward_out,backward_out],axis=2)
        if(self.isTrain == tf.convert_to_tensor(1)):
            print('isTrain --------------------')
            self.lstm_outputs = tf.nn.dropout(self.lstm_inputs,self.keep_prob)



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
            # self.sequence_lengths = np.array([310]*self.batch_size)
            # print(self.targets)
            print('Building  crf is use------')
            print(self.logits)
            print(self.targets)
            print(self.sequence_lengths)
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.targets, self.sequence_lengths)
            self.trans_params = trans_params
            self.loss = tf.reduce_mean(-log_likelihood)


    def build_optimizer(self):
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

    def train(self,batch_generator):
        # print('self.use_crf',self.use_crf)
        losses = []
        flag = 0
        if(flag == 1):
            sess = self.session
        else:
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

        step = 0
        for X, Y, M in batch_generator:
            step += 1
            feed, _ = self.getTrainfeed(X, M, Y)
            batch_loss, logits, _ ,_= sess.run([self.loss, self.logits, self.train_op,self.lstm_inputs], feed_dict=feed)

            print('loss', batch_loss)
            losses.append(batch_loss)
            if (step % 1000 == 0):
                print('asdasdasd')
                self.saver.save(sess, os.path.join('./crf_noOther_model/'), global_step=step)
                file = open('./resource/crf_losses4_train.txt', 'w', encoding='utf-8')
                file.write(str(losses))
                file.close()

    def predict(self,predict_generator):
        # self.isTrain = False
        predicts = []
        sess = self.session
        for X, M,Y in predict_generator:
            feed, lengths = self.getPredictfeed(X, M)
            if (self.use_crf == True):
                # inputs_embedding,other_embedding  = sess.run([self.inputs_embedding,self.other_embedding],feed_dict=feed)
                # print('inputs_embedding',inputs_embedding.shape)
                # print('other_embedding',other_embedding.shape)

                logits, trans_params = sess.run([self.logits, self.trans_params, ], feed_dict=feed)
                viterbi_sequences = []
                for logit, leng in zip(logits, lengths):
                    logit = logit[:leng]  # keep only the valid steps
                    viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                        logit, trans_params)
                    viterbi_sequences += [viterbi_seq]
                # print(viterbi_sequences)
                preds = [U.int_to_target(idx) for idx in list(viterbi_sequences[0])]
                predicts = predicts+preds
                # predicts.append(preds)
                real = [U.int_to_target(idx) for idx in list(Y[0])]

                print('=---------------------==')
                print('viterbi',preds)
                print('Y_lbale',real)
                print('=---------------------==')


            else:
                labels_pre = sess.run([self.labels_pre], feed_dict=feed)
                preds = [U.int_to_target(idx) for idx in list(labels_pre[0][0])]
                predicts.append(preds)
        return predicts
    def load(self, checkpoint):
        self.session = tf.Session()
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))

if __name__ == '__main__':
    pass
    g = batch_generator(200)
    rnn = MyMode(310,128,20,128,0.005,1723,47,128)
    # rnn.load(tf.train.latest_checkpoint('./crf_model/'))

    # rnn.use_crf = False
    # rnn.train(g)
    pre = predict_generator()
    # rnn.isTrain = 0

    rnn.load(tf.train.latest_checkpoint('./crf_noOther_model/'))
    predicts  = rnn.predict(pre)

    file = open('0-test.txt','r',encoding='utf-8')
    test_list=[]

    for line in file.readlines():
        line = line.strip()
        items = line.split()
        if(len(items) >2 ):
            test_list.append(items)
    test_predicts = []
    print(len(test_list),len(predicts))
    file2 = open('ceshi2.txt','w',encoding='utf-8')


    for test,pred in zip(test_list,predicts):
        test.append(pred)
        test_predicts.append(test)
        file2.write(test[0]+' '+test[1]+' '+test[2]+' '+pred+'\n')
    # print(test_list)
    file2.close()
    print(test_predicts)



