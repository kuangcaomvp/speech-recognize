#! /usr/bin/env python
# coding=utf-8

import tensorflow as tf
from config.config import cfg
from model.Transformer import Lm
"""
端到段模型冻结
"""
#模型保存路径
pb_file = "pb_save/model.pb"

#tensorflow 模型地址 checkpoint
ckpt_file = "logs_lm_tf/lm_train_loss=4.0981.ckpt-2"

# 输入输出节点
output_node_names = ["input_data", "label_in", 'wave_l',
                     'encode_samples/enc_num_blocks_5/multihead_attention_1/ln/add_1',#encode节点名称
                     'LogSoftmax',  #encode_log_softmax 节点名称
                     'strided_slice'] #decode_log_softmax 节点名称

feature_type = cfg.BASE.FEATURE_TYPE
data_last_channel = int(cfg.BASE.OUTPUT_NUM[feature_type])

input_data = tf.placeholder(tf.float32, shape=(None,None, data_last_channel),name='input_data')
label_in = tf.placeholder(tf.int32, shape=(None, None), name='label_in')
wave_l = tf.placeholder(tf.int32, shape=(None, 1), name='wave_l')

e2e = Lm(False)
encode, en_des = e2e.encode_network(input_data)
encode_log_softmax= tf.nn.log_softmax(en_des,axis=-1)

logits, preds = e2e.decode_network(encode, label_in,wave_l)
decode_log_softmax = tf.nn.log_softmax(logits)[:,-1]

sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
saver = tf.train.Saver()
saver.restore(sess, ckpt_file)

converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                            input_graph_def  = sess.graph.as_graph_def(),
                            output_node_names = output_node_names)

with tf.gfile.GFile(pb_file, "wb") as f:
    f.write(converted_graph_def.SerializeToString())




