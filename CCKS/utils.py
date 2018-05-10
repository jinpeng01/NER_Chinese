import numpy as np
import pickle
import os

max_length = 310
dir = './UtilsData'
class Utils:
    def __init__(self,filename='0-train.txt',vocab_save_file='vocab.pkl'):
        self.filename = filename
        self.vocab_save_file = vocab_save_file

        self.Buildvocab()
        self.BuildTarget_vocab()
        self.BuildOther_vocab()
        print(self.target_to_int_table)

        pass

    def Buildvocab(self):
        self.vocab = []
        if(os.path.exists('./UtilsData/word_to_int_table.txt') and os.path.exists('./UtilsData/int_to_word_table.txt')):
            word_to_int_table_file = open('./UtilsData/word_to_int_table.txt', 'r', encoding='utf-8')
            int_to_word_table_file = open('./UtilsData/int_to_word_table.txt', 'r', encoding='utf-8')
            self.word_to_int_table = eval(word_to_int_table_file.read())
            self.int_to_word_table = eval(int_to_word_table_file.read())
            word_to_int_table_file.close()
            int_to_word_table_file.close()
            self.vocab = self.word_to_int_table.keys()
        else:
            word_to_int_table_file = open('./UtilsData/word_to_int_table.txt','w',encoding='utf-8')
            int_to_word_table_file = open('./UtilsData/int_to_word_table.txt','w',encoding='utf-8')

            with open(self.filename, 'r', encoding='utf-8') as file:
                for line in file.readlines():
                    line = line.strip()
                    if (len(line) > 0):
                        items = line.split()
                        self.vocab.append(items[0])
            temp = set(self.vocab)
            self.vocab = list(temp)
            print('vocab.size', len(self.vocab))
            self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}
            self.int_to_word_table = dict(enumerate(self.vocab))
            s1 = str(self.word_to_int_table)
            s2 = str(self.int_to_word_table)
            word_to_int_table_file.write(s1)
            int_to_word_table_file.write(s2)
            word_to_int_table_file.close()
            int_to_word_table_file.close()

    def BuildTarget_vocab(self):
        self.target_vocab = []
        if (os.path.exists('./UtilsData/target_to_int_table.txt') and os.path.exists('./UtilsData/int_to_target_table.txt')):
            target_to_int_table_file = open('./UtilsData/target_to_int_table.txt', 'r', encoding='utf-8')
            int_to_target_table_file = open('./UtilsData/int_to_target_table.txt', 'r', encoding='utf-8')
            self.target_to_int_table = eval(target_to_int_table_file.read())
            self.int_to_target_table = eval(int_to_target_table_file.read())
            target_to_int_table_file.close()
            int_to_target_table_file.close()
            self.target_vocab = self.target_to_int_table.keys()

        else:
            target_to_int_table_file = open('./UtilsData/target_to_int_table.txt', 'w', encoding='utf-8')
            int_to_target_table_file = open('./UtilsData/int_to_target_table.txt', 'w', encoding='utf-8')
            with open(self.filename, 'r', encoding='utf-8') as file:
                for line in file.readlines():
                    line = line.strip()
                    if (len(line) > 2):
                        items = line.split()
                        self.target_vocab.append(items[2])

            temp = set(self.target_vocab)
            self.target_vocab = list(temp)
            # print('target_vocab.size', len(self.target_vocab))

            self.target_to_int_table = {c: i for i, c in enumerate(self.target_vocab)}
            self.int_to_target_table = dict(enumerate(self.target_vocab))

            s1 = str(self.target_to_int_table)
            s2 = str(self.int_to_target_table)
            target_to_int_table_file.write(s1)
            int_to_target_table_file.write(s2)
            target_to_int_table_file.close()
            int_to_target_table_file.close()

    def BuildOther_vocab(self):
        self.other_vocab = []
        if (os.path.exists('./UtilsData/other_to_int_table.txt') and os.path.exists('./UtilsData/int_to_other_table.txt')):
            other_to_int_table_file = open('./UtilsData/other_to_int_table.txt', 'r', encoding='utf-8')
            int_to_other_table_file = open('./UtilsData/int_to_other_table.txt', 'r', encoding='utf-8')
            self.other_to_int_table = eval(other_to_int_table_file.read())
            self.int_to_other_table = eval(int_to_other_table_file.read())
            other_to_int_table_file.close()
            int_to_other_table_file.close()
            self.other_vocab = self.other_to_int_table.keys()
        else:
            other_to_int_table_file = open('./UtilsData/other_to_int_table.txt', 'w', encoding='utf-8')
            int_to_other_table_file = open('./UtilsData/int_to_other_table.txt', 'w', encoding='utf-8')
            with open(self.filename, 'r', encoding='utf-8') as file:
                for line in file.readlines():
                    line = line.strip()
                    if (len(line) > 2):
                        items = line.split()
                        self.other_vocab.append(items[1])


                temp = set(self.other_vocab)
                self.other_vocab = list(temp)
                print('other_vocab.size', len(self.other_vocab))
                self.other_to_int_table = {c: i for i, c in enumerate(self.other_vocab)}
                # print(self.other_to_int_table['NNB'])
                self.int_to_other_table = dict(enumerate(self.other_vocab))

                s1 = str(self.other_to_int_table)
                s2 = str(self.int_to_other_table)
                other_to_int_table_file.write(s1)
                int_to_other_table_file.write(s2)
                other_to_int_table_file.close()
                int_to_other_table_file.close()

    def vocab_size(self):
        return len(self.vocab)

    def target_voacab_size(self):
        return len(self.target_vocab)

    def other_size(self):
        return len(self.other_vocab)

    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return len(self.vocab)

    def int_to_word(self, index):
        if index == len(self.vocab):
            return '<unk>'
        elif index < len(self.vocab):
            return self.int_to_word_table[index]
        else:
            raise Exception('Unknown index!')

    def target_to_int(self,target):
        if target in self.target_to_int_table:
            return self.target_to_int_table[target]
        else:
            return len(self.target_vocab)

    def int_to_target(self,index):
        if index == len(self.target_vocab):
            return '<unk>'
        elif index < len(self.target_vocab):
            return self.int_to_target_table[index]
        elif index == len(self.target_vocab)+1 or index == len(self.target_vocab)+2:
            return 'None'
        else:
            # raise Exception('Unknown index!')
            return 'I do not know completely'
    def other_to_int(self, other):

        if other in self.other_to_int_table:
            return self.other_to_int_table[other]
        else:
            return len(self.other_vocab)

    def int_to_other(self, index):
        if index == len(self.other_vocab):
            return '<unk>'
        elif index < len(self.other_vocab):
            return self.int_to_other_table[index]
        else:
            raise Exception('Unknown index!')

    # def text_to_arr(self, text):
    #     arr = []
    #     for word in text:
    #         arr.append(self.word_to_int(word))
    #     return np.array(arr)
    #
    # def arr_to_text(self, arr):
    #     file = open('data/arrtotext.txt','w+',encoding='utf-8')
    #     words = []
    #     for index in arr:
    #         words.append(self.int_to_word(index))
    #     return "".join(words)
    #
    # def other_to_arr(self, other):
    #     arr = []
    #     for word in other:
    #         arr.append(self.other_to_int(word))
    #     return np.array(arr)
    #
    # def arr_to_other(self, arr):
    #     # file = open('data/arrtotext.txt', 'w+', encoding='utf-8')
    #     words = []
    #     for index in arr:
    #         words.append(self.int_to_other(index))
    #     return "".join(words)
    #
    # def target_to_arr(self, target):
    #     arr = []
    #     for word in target:
    #         arr.append(self.target_to_int(word))
    #     return np.array(arr)
    #
    # def arr_to_target(self, target):
    #     # file = open('data/arrtotext.txt', 'w+', encoding='utf-8')
    #     words = []
    #     for index in target:
    #         words.append(self.int_to_target(index))
    #     return "".join(words)
    def toIndex(self,_list,table):
        indexs = []
        for item in _list:
            if(item in table):
                indexs.append(table[item])
            else:
                indexs.append(len(table))
        return indexs


    def padding(self,sequence,pad_token,max_length):

        leng = len(sequence)
        if(len(sequence) < max_length):
            sequence = sequence+([pad_token]*(max_length-len(sequence)))
        # else
        #     sequence = sequence+([pad_token]*(max_length-len(sequence)))

        return sequence,leng
        pass

    def paddingList(self,sequence_list,pad_token):

        sequences = []
        lengths=[]
        max_length = max(map(lambda x:len(x),sequence_list))
        for item in sequence_list:
            sequence,realleng =  self.padding(item,pad_token,max_length)
            sequences.append(sequence)
            lengths = lengths+[realleng]

        return sequences,lengths

    def save_to_file(self, vocab_save_file):
        with open(vocab_save_file, 'wb') as f:
            pickle.dump(self.vocab, f)

def batch_generator(batch_size,fileName = '0-train.txt'):
    U = Utils()
    word_list_item=[]
    other_feature_list_item = []
    target_list_item = []

    word_index_vector = []
    other_feature_index_vector = []
    target_index_vector = []

    with open(fileName, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            items = line.split()
            if(len(items)>2):
                word_list_item.append(items[0])
                other_feature_list_item.append(items[1])
                target_list_item.append(items[2])
            else:


                word_index_vector_item = U.toIndex(word_list_item,U.word_to_int_table)
                word_index_vector.append(word_index_vector_item)

                other_feature_index_vector_item = U.toIndex(other_feature_list_item,U.other_to_int_table)
                other_feature_index_vector.append(other_feature_index_vector_item)

                target_index_vector_item = U.toIndex(target_list_item,U.target_to_int_table)
                target_index_vector.append(target_index_vector_item)


                word_list_item=[]
                other_feature_list_item=[]
                target_list_item=[]
    while(True):
        startindex = 0
        while (startindex <len(target_index_vector)):
            X = word_index_vector[startindex:startindex + batch_size]
            Y = target_index_vector[startindex:startindex + batch_size]
            M = other_feature_index_vector[startindex:startindex + batch_size]
            startindex = startindex + batch_size

            yield X,Y,M



def predict_generator(fileName = '0-test.txt'):
    U = Utils()
    word_list_item = []
    other_feature_list_item = []
    target_list_item = []

    word_index_vector = []
    other_feature_index_vector = []
    target_index_vector = []

    with open(fileName, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            items = line.split()

            if (len(items) > 2):
                word_list_item.append(items[0])
                other_feature_list_item.append(items[1])
                target_list_item.append(items[2])
            else:
                if(len(word_list_item)!= len(other_feature_list_item)):
                    print('word_list_item',word_list_item)
                    print('other_feature',other_feature_list_item)
                word_index_vector_item = U.toIndex(word_list_item, U.word_to_int_table)
                word_index_vector.append(word_index_vector_item)

                other_feature_index_vector_item = U.toIndex(other_feature_list_item, U.other_to_int_table)
                other_feature_index_vector.append(other_feature_index_vector_item)

                target_index_vector_item = U.toIndex(target_list_item, U.target_to_int_table)
                target_index_vector.append(target_index_vector_item)
                if (len(word_index_vector_item) != len(other_feature_index_vector_item)):
                    print('word_index_vector_item', word_list_item)
                    print('other_feature_index_vector_item', other_feature_list_item)
                word_list_item = []
                other_feature_list_item = []
                target_list_item = []

    startindex = 0
    batch_size=1
    while (startindex < len(other_feature_index_vector)):
        X = word_index_vector[startindex:startindex + batch_size]
        M = other_feature_index_vector[startindex:startindex + batch_size]
        Y = target_index_vector[startindex:startindex + batch_size]
        startindex = startindex + batch_size
        # if(len(X[0]) != len(M[0])):
            # print("X",X)
            # print("M",M)



        yield X,M,Y
















# U = Utils()
# batch_generator(20,'all.txt')

# s = U.padding([1,2,3],4,2)
# print(s)

predict_generator()







