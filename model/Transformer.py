import math

import numpy as np
import tensorflow as tf

from config.config import cfg


def normalize(inputs,
              epsilon = 1e-8,
              scope="ln",
              reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable(name='beta', shape=params_shape, initializer=tf.constant_initializer(0.0))
        gamma = tf.get_variable(name='gamma', shape=params_shape, initializer=tf.constant_initializer(1.0))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta

    return outputs



def embedding(vocab_size,
              num_units,
              zero_pad=False,
              scope="embedding",
              reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)

    return lookup_table


def creat_pe(max_length,d_model):
    pe = np.zeros((max_length, d_model))
    position = np.expand_dims(np.arange(0, max_length, dtype=np.float32),axis=1)
    div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) *
                         -(math.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


def mask(inputs, key_masks):
        padding_num = -2 ** 32 + 1
        key_masks = tf.tile(key_masks, [tf.shape(inputs)[0] // tf.shape(key_masks)[0], tf.shape(inputs)[1] // tf.shape(key_masks)[1], 1])  # (h*N, t1, t2)
        key_masks = tf.to_float(key_masks)
        paddings = tf.ones_like(key_masks) * padding_num
        outputs = tf.where(tf.equal(key_masks, 0), paddings, inputs)
        outputs = tf.nn.softmax(outputs)
        paddings_zeros = tf.zeros_like(key_masks)
        outputs = tf.where(tf.equal(key_masks, 0), paddings_zeros, outputs)
        return outputs

def scaled_dot_product_attention(Q, K, V,key_masks, dropout_rate=0.,
                                 training=True,
                                 scope="scaled_dot_product_attention"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]

        # dot product
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

        # scale
        outputs /= d_k ** 0.5

        # key masking
        outputs = mask(outputs, key_masks=key_masks)

        # dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

        # weighted sum (context vectors)
        outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

        return outputs

def positional_encoding(inputs,
                        position_enc,
                        masking=False,
                        scope="positional_encoding"):
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1] # dynamic
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) # (N, T)
        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)
        # masks
        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

    return tf.to_float(outputs)



def multihead_attention(queries, keys, values,key_masks,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        scope="multihead_attention"):

    d_model = queries.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Linear projections
        Q = tf.layers.dense(queries, d_model, use_bias=True)  # (N, T_q, d_model)
        K = tf.layers.dense(keys, d_model, use_bias=True)  # (N, T_k, d_model)
        V = tf.layers.dense(values, d_model, use_bias=True)  # (N, T_k, d_model)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)

        # Attention
        outputs = scaled_dot_product_attention(Q_, K_, V_, key_masks, dropout_rate, is_training)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, d_model)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = normalize(outputs) # (N, T_q, C)


        return outputs

    

def feedforward(inputs, 
                num_units=[2048, 512],
                scope="multihead_attention",
                reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        
        # Residual connection
        outputs += inputs
        
        # Normalize
        outputs = normalize(outputs)
    
    return outputs


def label_smoothing(inputs, epsilon=0.1):
    K = inputs.get_shape().as_list()[-1] # number of channels
    return ((1-epsilon) * inputs) + (epsilon / K)



def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names



def label_mask_tril(x):
    ys_mask = tf.math.not_equal(x, -1)
    size = tf.shape(ys_mask)[-1]
    ret = tf.ones(shape=[size, size], dtype=tf.float32)
    tril = tf.expand_dims(tf.linalg.LinearOperatorLowerTriangular(ret).to_dense(), axis=0)
    tril = tf.to_int32(tril)
    ys_mask_exp = tf.expand_dims(ys_mask, axis=-2)
    ys_mask_exp = tf.to_int32(ys_mask_exp)
    c = tf.bitwise.bitwise_and(ys_mask_exp, tril)
    return c



class Lm(object):
    def __init__(self, is_training, dropout_rate_position=0.0):
            self.is_training = is_training
            self.hidden_units = 256
            self.label_vocab_size = len(read_class_names(cfg.BASE.DICT))
            self.num_heads = 4
            self.en_num_blocks = 6
            self.de_num_blocks = 6
            self.max_length = 100
            self.dropout_rate_position = dropout_rate_position

            self.dropout_rate = dropout_rate_position

            self.embbeding = embedding(vocab_size=self.label_vocab_size, num_units=self.hidden_units)
            self.en_pe = tf.convert_to_tensor(creat_pe(1000, self.hidden_units), tf.float32)
            self.pe = tf.convert_to_tensor(creat_pe(self.max_length, self.hidden_units), tf.float32)


    def encode_network(self,x, wave_len = None):
        with tf.variable_scope('encode_samples', reuse=tf.AUTO_REUSE):
            x = tf.expand_dims(x, -1)
            emb = tf.layers.conv2d(x, self.hidden_units, 3, 2, activation=tf.nn.relu)
            emb = tf.layers.conv2d(emb, self.hidden_units, 3, 2, activation=tf.nn.relu)

            emb = tf.reshape(emb, [tf.shape(emb)[0], tf.shape(emb)[1], self.hidden_units * 9])
            emb = tf.layers.dense(emb, self.hidden_units, activation=tf.nn.relu)
            enc = emb * (self.hidden_units**0.5) + positional_encoding(emb, self.en_pe)
            enc = tf.layers.dropout(enc,
                                    rate=self.dropout_rate_position,
                                    training=tf.convert_to_tensor(self.is_training))

            seq_range = tf.range(0, tf.shape(enc)[1], dtype=tf.int32)
            seq_range = tf.expand_dims(seq_range, axis=0)
            seq_range = tf.tile(seq_range, [tf.shape(x)[0], 1])
            if wave_len is None:
                src_masks = seq_range <= seq_range
            else:
                src_masks = seq_range <= wave_len
            src_masks = tf.expand_dims(src_masks, axis=1)

            for i in range(self.en_num_blocks):
                with tf.variable_scope("enc_num_blocks_{}".format(i)):
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              key_masks=src_masks,
                                              num_heads=self.num_heads,
                                              dropout_rate=self.dropout_rate,
                                              is_training=self.is_training)

                    enc = feedforward(enc, num_units=[4 * self.hidden_units, self.hidden_units])
            en_des = tf.layers.dense(enc, self.label_vocab_size)
        return enc,en_des



    def decode_network(self,encode,y,wave_len):
        with tf.variable_scope('decode_samples', reuse=tf.AUTO_REUSE):
            tgt_masks = label_mask_tril(y)

            seq_range = tf.range(0, tf.shape(encode)[1], dtype=tf.int32)
            seq_range = tf.expand_dims(seq_range, axis=0)
            seq_range = tf.tile(seq_range, [tf.shape(wave_len)[0], 1])
            src_masks = seq_range <= wave_len
            src_masks = tf.expand_dims(src_masks, axis=1)

            # embedding
            emb = tf.nn.embedding_lookup(self.embbeding, y)

            dec = emb * (self.hidden_units**0.5) + positional_encoding(emb, self.pe)

            ## layer_nomal and Dropout
            dec = normalize(dec)
            dec = tf.layers.dropout(dec,
                                        rate=self.dropout_rate_position,
                                        training=tf.convert_to_tensor(self.is_training))
                        
            ## Blocks
            for i in range(self.de_num_blocks):
                with tf.variable_scope("dec_num_blocks_{}".format(i)):
                    ### Multihead Attention
                    dec = multihead_attention(queries=dec,
                                                    keys=dec,
                                                    values=dec,
                                                    key_masks=tgt_masks,
                                                    num_heads=self.num_heads,
                                                    dropout_rate=self.dropout_rate,
                                                    is_training=self.is_training)

                    dec = multihead_attention(queries=dec,
                                              keys=encode,
                                              values=encode,
                                              key_masks=src_masks,
                                              num_heads=self.num_heads,
                                              dropout_rate=self.dropout_rate,
                                              is_training=self.is_training)
                                
                    ### Feed Forward
                    dec = feedforward(dec, num_units=[4*self.hidden_units, self.hidden_units])
                                    
            # Final linear projection
            logits = tf.layers.dense(dec, self.label_vocab_size,name='dense_output')
            preds = tf.to_int32(tf.argmax(logits, axis=-1))

        return logits,preds

    def compute_acc(self,preds,y):
        with tf.name_scope(name='acc'):
            istarget = tf.to_float(tf.not_equal(y, -1))
            acc = tf.reduce_sum(tf.to_float(tf.equal(preds, y))*istarget)/ (tf.reduce_sum(istarget))
        return acc

    def compute_loss(self,logits,y):
        # Loss
        with tf.name_scope(name='loss'):
            y_smoothed = label_smoothing(tf.one_hot(y, depth=self.label_vocab_size))
            istarget = tf.to_float(tf.not_equal(y, -1))
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_smoothed)
            mean_loss = tf.reduce_sum(loss*istarget) / (tf.reduce_sum(istarget))
        return mean_loss


    def compute_loss_1(self, logits, y, label_smoothing=0.1):
        with tf.name_scope(name='loss'):
            confidence = 1.0 - label_smoothing
            low_confidence = label_smoothing / \
                             tf.cast(self.label_vocab_size - 1, dtype=tf.float32)
            soft_targets = tf.one_hot(
                tf.cast(y, tf.int32),
                depth=self.label_vocab_size,
                on_value=confidence,
                off_value=low_confidence,
            )
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=soft_targets)
            normalizing_constant = -(
                    confidence * tf.log(confidence) +
                    tf.cast(self.label_vocab_size - 1, dtype=tf.float32) *
                    low_confidence * tf.log(low_confidence + 1e-20)
            )
            loss -= normalizing_constant
            istarget = tf.to_float(tf.not_equal(y, -1))
            mean_loss = tf.reduce_sum(loss * istarget) / (tf.reduce_sum(istarget))
        return mean_loss

    def compute_ctc_loss(self,y,input_lengths,label_lengths,encode_softmax):
        with tf.name_scope(name='cts_loss'):
            ctc_loss = tf.keras.backend.ctc_batch_cost(y_true=y,y_pred=encode_softmax
                                        ,input_length=input_lengths,label_length=label_lengths)
            batch_loss = tf.reduce_mean(ctc_loss, name='batch_loss')
        return batch_loss
