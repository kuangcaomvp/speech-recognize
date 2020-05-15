#! /usr/bin/env python
# coding=utf-8

import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from tqdm import tqdm

from config.config import cfg
from model.Transformer import Lm
from generate_data import get_data

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def get_available_gpus():
    r"""
    Returns the number of GPUs available on this system.
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']



def calculate_mean_edit_distance_and_loss(iterator,dropout_rate_position,trainable):
    r'''
    This routine beam search decodes a mini-batch and calculates the loss and mean edit distance.
    Next to total and average loss it returns the mean edit distance,
    the decoded result and the batch's original Y.
    '''
    # Obtain the next batch of data
    wave, wave_len, label, label_length, label_in, label_out, _ = iterator.get_next()
    model = Lm(trainable, dropout_rate_position)

    encode, en_des = model.encode_network(wave,wave_len)
    logits, preds = model.decode_network(encode, label_in,wave_len)

    att_loss = model.compute_loss_1(logits,label_out)
    acc = model.compute_acc(preds,label_out)
    encode_softmax = tf.nn.softmax(en_des, axis=-1)
    ctc_loss = model.compute_ctc_loss(label, wave_len, label_length, encode_softmax)
    loss = ctc_loss * 0.3 + (1 - 0.3) * att_loss
    return loss,att_loss,acc,ctc_loss

def get_tower_results(iterator, optimizer,dropout_rate_position,trainable):
    # To calculate the mean of the losses
    tower_avg_losses = []
    tower_avg_att_losses = []
    tower_avg_ctc_losses = []
    tower_avg_acc = []
    # Tower gradients to return
    tower_gradients = []

    available_devices=get_available_gpus()
    with tf.variable_scope(tf.get_variable_scope()):
        # Loop over available_devices
        for i in range(len(available_devices)):
            # Execute operations of tower i on device i
            device = get_available_gpus()[i]
            with tf.device(device):
                # Create a scope for all operations of tower i
                with tf.name_scope('tower_%d' % i):
                    loss, att_loss, acc, ctc_loss = calculate_mean_edit_distance_and_loss(iterator,dropout_rate_position,trainable)
                    tf.get_variable_scope().reuse_variables()

                    tower_avg_losses.append(loss)
                    gradients = optimizer.compute_gradients(loss)
                    tower_gradients.append(gradients)
                    tower_avg_att_losses.append(att_loss)
                    tower_avg_ctc_losses.append(ctc_loss)
                    tower_avg_acc.append(acc)
    avg_loss = tf.reduce_mean(input_tensor=tower_avg_losses, axis=0)
    avg_att = tf.reduce_mean(input_tensor=tower_avg_att_losses, axis=0)
    avg_ctc = tf.reduce_mean(input_tensor=tower_avg_ctc_losses, axis=0)
    avg_acc = tf.reduce_mean(input_tensor=tower_avg_acc, axis=0)
    return tower_gradients, avg_loss, avg_att, avg_ctc, avg_acc

def average_gradients(tower_gradients):
    r'''
    A routine for computing each variable's average of the gradients obtained from the GPUs.
    Note also that this code acts as a synchronization point as it requires all
    GPUs to be finished with their mini-batch before it can run to completion.
    '''
    # List of average gradients to return to the caller
    average_grads = []

    # Run this on cpu_device to conserve GPU memory
    with tf.device('/cpu:0'):
        # Loop over gradient/variable pairs from all towers
        for grad_and_vars in zip(*tower_gradients):
            # Introduce grads to store the gradients for the current variable
            grads = []

            # Loop over the gradients for the current variable
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(input_tensor=grad, axis=0)

            # Create a gradient/variable tuple for the current variable with its average gradient
            grad_and_var = (grad, grad_and_vars[0][1])

            # Add the current tuple to average_grads
            average_grads.append(grad_and_var)

    # Return result to caller
    return average_grads

def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

def train():
    trainset = get_data('train')
    testset = get_data('dev')

    global_step = tf.train.get_or_create_global_step()

    steps_per_period=len(trainset)
    warmup_steps = tf.constant(cfg.BASE.warmup_periods * steps_per_period,
                               dtype=tf.float32, name='warmup_steps')
    train_steps = tf.constant(cfg.BASE.epochs * steps_per_period,
                              dtype=tf.float32, name='train_steps')

    warmup_lr = tf.to_float(global_step) / tf.to_float(warmup_steps) \
                * cfg.BASE.lr

    decay_lr = tf.train.cosine_decay(
        cfg.BASE.lr,
        global_step=tf.to_float(global_step) - warmup_steps,
        decay_steps=train_steps - warmup_steps,
        alpha=0.01)
    learn_rate = tf.where(tf.to_float(global_step) < warmup_steps, warmup_lr, decay_lr)
    optimizer = tf.train.AdamOptimizer(learn_rate)

    iterator = tf.data.Iterator.from_structure(trainset.dataset.output_types,
                                               trainset.dataset.output_shapes,
                                               output_classes=trainset.dataset.output_classes)
    train_init_op = iterator.make_initializer(trainset.dataset)
    test_init_op = iterator.make_initializer(testset.dataset)

    trainable = tf.placeholder(dtype=tf.bool, name='training')
    dropout_rate_position = tf.placeholder(tf.float32, shape=(), name='drop_out')
    gradients, loss, att, ctc, acc = get_tower_results(iterator, optimizer,dropout_rate_position,trainable)

    avg_tower_gradients = average_gradients(gradients)

    grads, all_vars = zip(*avg_tower_gradients)
    clipped, gnorm = tf.clip_by_global_norm(grads, 0.25)
    grads_and_vars = list(zip(clipped, all_vars))

    apply_gradient_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)



    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        loader = tf.train.Saver(tf.global_variables())
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        sess.run(tf.global_variables_initializer())
        try:
            print('=> Restoring weights from: %s ... ' % cfg.BASE.initial_weight)
            loader.restore(sess, cfg.BASE.initial_weight)
        except:
            print('=> %s does not exist !!!' % cfg.BASE.initial_weight)
            print('=> Now it starts to train LM from scratch ...')


        for epoch in range(1, 1+cfg.BASE.epochs):
            #初始化数据集
            sess.run(train_init_op)
            train_epoch_loss, test_epoch_loss = [], []
            train_epoch_acc, test_epoch_acc = [], []
            train_epoch_ctc, test_epoch_ctc = [], []
            train_epoch_att, test_epoch_att = [], []
            pbar = tqdm(range(len(trainset)+1))
            for i in pbar:
                try:
                    _, train_step_loss,train_step_acc,train_step_ctc,train_step_att, \
                    global_step_val = sess.run(
                        [apply_gradient_op, loss,acc,ctc,att,
                         global_step],feed_dict={
                                                    trainable:    True,
                                                    dropout_rate_position:0.1

                    })

                    train_epoch_loss.append(train_step_loss)
                    train_epoch_acc.append(train_step_acc)
                    train_epoch_ctc.append(train_step_ctc)
                    train_epoch_att.append(train_step_att)
                    pbar.set_description("loss:%.2f" % train_step_loss)
                    pbar.set_postfix(acc=train_step_acc)
                except tf.errors.OutOfRangeError:
                    break


            sess.run(test_init_op)
            while True:
                try:

                    test_step_loss,test_step_acc,test_step_ctc,test_step_att = sess.run( [loss,acc,ctc,att], feed_dict={
                                                    trainable:    False,
                                                    dropout_rate_position:0.0
                    })

                    test_epoch_loss.append(test_step_loss)
                    test_epoch_acc.append(test_step_acc)
                    test_epoch_ctc.append(test_step_ctc)
                    test_epoch_att.append(test_step_att)

                except tf.errors.OutOfRangeError:
                    break
            train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)
            train_epoch_ctc,test_epoch_ctc = np.mean(train_epoch_ctc), np.mean(test_epoch_ctc)
            train_epoch_att, test_epoch_att = np.mean(train_epoch_att), np.mean(test_epoch_att)

            train_epoch_acc, test_epoch_acc = np.mean(train_epoch_acc), np.mean(test_epoch_acc)
            ckpt_file = "./logs_lm_tf/lm_train_loss=%.4f.ckpt" % train_epoch_loss
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("=> Epoch: %2d Time: %s Tr_loss: %.2f  Tr_acc: %.2f Te_acc: %.2f "
                  " tr_ctc: %.2f tr_att: %.2f "
                  % (epoch, log_time, train_epoch_loss, train_epoch_acc, test_epoch_acc,
                     train_epoch_ctc,  train_epoch_att))
            saver.save(sess, ckpt_file, global_step=epoch)




if __name__ == '__main__':
    train()




