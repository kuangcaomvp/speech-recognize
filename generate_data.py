import os
import time
from random import shuffle

import numpy as np
import tensorflow as tf

from config.config import cfg
from util.feature import load_sample


class get_data(object):
    def __init__(self, train_type):
        # #训练方式  train test dev
        self.data_type = train_type
        #数据类型  mfcc 或mel
        self.feature_type = cfg.BASE.FEATURE_TYPE
        self.nomalnize= cfg.BASE.NOMALNIZE

        # 采样特征的最后通道数
        self.data_last_channel= int(cfg.BASE.OUTPUT_NUM[self.feature_type])
        # 批次数
        self.batch_size = cfg.TRAIN.BATCH_SIZE if train_type == 'train' else cfg.TEST.BATCH_SIZE
        # 文件基础目录
        self.data_path = cfg.BASE.DATA_PATH
        # 需要加载的数据集 在config下配置
        self.read_files = cfg.TRAIN.read_files if train_type == 'train' else cfg.TEST.read_files

        # record 目录

        self.sos  = '<EOS>'
        self.eos  = '<EOS>'

        #字典  根据数据集自己修改 训练与transformer.py保持一致
        self.han_vocab = self.readtxt(cfg.BASE.DICT)
        self.source_init()
        self.dataset = tf.data.Dataset.from_generator(self.generate_values,
                                              (tf.string, tf.int32,tf.int32,tf.int32)) \
                                        .map(self.wrapped_complex_calulation,
                                             num_parallel_calls=8) \
                                        .filter(self.filter_fn)          \
                                        .padded_batch(self.batch_size,
                                        padded_shapes=([None, self.data_last_channel], [None], [None], [None], [None], [None], [None]),
                                        padding_values=(0.0,-1,-1,-1,-1,-1,-1))\
                                        .prefetch(8)


        self.iterator = self.dataset.make_initializable_iterator()
        self.next_element = self.iterator.get_next()
        self.init_op = self.iterator.initializer

    def filter_fn(self,data,wave_len,label,label_len,label_in,label_out,ctc_len):

        return  tf.greater(1000,wave_len) & tf.greater(wave_len,ctc_len)


    def wrapped_complex_calulation(self,wave,label,label_in,label_out):
        data,data_len,label,label_len,\
        label_in,label_out,ctc_len= \
            tf.py_func(func=self.calculation,
                          inp=(wave,label,label_in,label_out),
                          Tout=(tf.double, tf.int64, tf.int32, tf.int64, tf.int32, tf.int32, tf.int64))

        data=tf.cast(data,tf.float32)

        data_len=[tf.shape(data)[0]//4-1]
        data_len= tf.cast(data_len,tf.int32)
        label = tf.cast(label, tf.int32)
        label_len = tf.cast(label_len, tf.int32)
        label_in = tf.cast(label_in, tf.int32)
        label_out = tf.cast(label_out, tf.int32)
        ctc_len= tf.cast(ctc_len, tf.int32)
        return data,data_len,label,label_len,label_in,label_out,ctc_len


    def calculation(self,wave,label,label_in,label_out):
        data_input,wave_len =load_sample(wave,self.feature_type,self.nomalnize,self.data_type)

        if data_input.shape[0] % 4 != 0:
            data = np.zeros((data_input.shape[0] // 4 * 4 + 4, data_input.shape[1]))
            data[:data_input.shape[0], :] = data_input
        else:
            data = data_input

        label_len=len(label)
        data = data.astype(np.float)
        ctc_len = self.ctc_len(label)
        return data,[wave_len//4-1],label,[label_len],label_in,label_out,[ctc_len]

    def source_init(self):
        print('get source list...')
        self.wav_lst = []
        self.han_lst = []
        for file in self.read_files:
            print('load ', file, ' data...')
            sub_file = 'data/' + file
            with open(sub_file, 'r', encoding='utf8') as f:
                data = f.readlines()
            shuffle(data)
            for line in data:
                wav_file,  han = line.strip().split('\t')
                han = han.lower()
                try:
                    label = [self.han_vocab.index(i.lower()) for i in han.strip().split(' ')]
                    if len(label) >= 100:
                        continue

                    npy_path = os.path.join(self.data_path, wav_file)
                    self.wav_lst.append(npy_path)

                    self.han_lst.append(label)
                except:
                    print(wav_file,han)

            print('leng:'+str(len(self.han_lst)))

        self.batch_num = len(self.han_lst) // self.batch_size

    def __len__(self):
        return self.batch_num

    def __iter__(self):
        return self


    def generate_values(self):
        SOS = [self.han_vocab.index(self.sos)]
        EOS = [self.han_vocab.index(self.eos)]
        n_= [i for i in range(len(self.han_lst))]
        shuffle(n_)
        for j in n_:
            wave = self.wav_lst[j]
            label = self.han_lst[j]
            label_in = SOS + label
            label_out = label + EOS
            yield  wave,label,label_in,label_out


    def ctc_len(self, label):
        add_len = 0
        label_len = len(label)
        for i in range(label_len - 1):
            if label[i] == label[i + 1]:
                add_len += 1
        return label_len + add_len



    def readtxt(self,txt):
        with open(txt,'r') as f:
            lines = f.readlines()
        return [line.strip() for line in lines]


if __name__=='__main__':
    testset = get_data('train')
    sess = tf.InteractiveSession()
    count = 0
    sess.run(testset.init_op)
    t1= time.time()
    while True:
        try:
            wave,wave_len, label,label_length, label_in \
                , label_out,_ \
             = sess.run(testset.next_element)
            print(wave)
            break
            # print(i)

        except tf.errors.OutOfRangeError:
            # print("End of dataset")
            break
    t2 = time.time()
    print(t2-t1)





