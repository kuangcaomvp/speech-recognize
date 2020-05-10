#coding=utf-8


import numpy as np
import soundfile
import tensorflow as tf

from config.config import cfg
from ctc_prefix_score import CTCPrefixScore


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def readtxt(txt):
    with open(txt, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

def read_pb_return_tensors(graph, pb_file, return_elements):

    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        frozen_graph_def = tf.GraphDef()
        frozen_graph_def.ParseFromString(f.read())

    with graph.as_default():
        return_elements = tf.import_graph_def(frozen_graph_def,
                                              return_elements=return_elements)
    return return_elements

def end_detect(ended_hyps, i, M=3, D_end=np.log(1 * np.exp(-10))):
    if len(ended_hyps) == 0:
        return False

    count = 0
    best_hyp = sorted(ended_hyps, key=lambda x: x['score'], reverse=True)[0]
    for m in range(M):
        # get ended_hyps with their length is i - m
        hyp_length = i - m
        hyps_same_length = [x for x in ended_hyps if len(x['yseq']) == hyp_length]
        if len(hyps_same_length) > 0:
            best_hyp_same_length = sorted(hyps_same_length, key=lambda x: x['score'], reverse=True)[0]
            if best_hyp_same_length['score'] - best_hyp['score'] < D_end:
                count += 1

    if count == M:
        return True
    else:
        return False


def beam_search_decode_with_ctc(x, han_vocab, sess):
    enc, lpz = sess.run([return_tensors[3], return_tensors[4]], {return_tensors[0]: np.array([x])})
    # print(enc.shape)
    lpz = np.squeeze(lpz, axis=0)

    if ctc_weight == 0.0:
        lpz =None

    maxlen = enc.shape[1]-1
    minlen = 0
    eos = han_vocab.index('<EOS>')
    hyp = {'score': 0.0, 'yseq': [eos]}
    if lpz is not None:
        ctc_prefix_score = CTCPrefixScore(lpz, han_vocab.index('*'), eos)
        hyp['ctc_state_prev'] = ctc_prefix_score.initial_state()
        hyp['ctc_score_prev'] = 0.0
        if ctc_weight != 1.0:
            ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
        else:
            ctc_beam = lpz.shape[-1]

    hyps = [hyp]
    ended_hyps = []

    for i in range(maxlen):
        hyps_best_kept = []
        max_len = max(len(hyp['yseq']) for hyp in hyps)
        ys = np.ones(shape=(len(hyps),max_len))*eos
        for k, hyp in enumerate(hyps):
            ys[k, :len(hyp['yseq'])] =hyp['yseq']

        local_att_scores_all = sess.run(return_tensors[5], {return_tensors[3]: np.tile(enc,(len(hyps),1,1)),
                                                            return_tensors[1]: ys,
                                                            return_tensors[2]: np.tile(np.array([[maxlen]]),(len(hyps),1))})

        local_scores_all = local_att_scores_all

        # **  ***    **
        if lpz is not None:
            local_best_scores_all, local_best_ids_all = sess.run([score, ids],
                                                                 {topk_input: local_att_scores_all,
                                                                  beam_size: ctc_beam})

            ctc_state_prev = np.array([hyp['ctc_state_prev'] for hyp in hyps])
            ctc_score_prev = np.array([hyp['ctc_score_prev'] for hyp in hyps])
            ctc_score_prev = ctc_score_prev[:, np.newaxis]

            ctc_scores_all, ctc_states_all = ctc_prefix_score(
                    ys, local_best_ids_all, ctc_state_prev)

            # 取local_best_ids_all对应的分数
            local_att_scores_all_now = local_att_scores_all[
                np.arange(0, local_best_ids_all.shape[0])[:, np.newaxis], local_best_ids_all]

            local_scores_all = (1.0 - ctc_weight) * local_att_scores_all_now \
                               + ctc_weight * (ctc_scores_all - ctc_score_prev)

            local_best_scores_all, joint_best_ids_all = sess.run([score, ids],
                                                                 {topk_input: local_scores_all, beam_size: beam})
            local_best_ids_all = local_best_ids_all[
                np.arange(0, joint_best_ids_all.shape[0])[:, np.newaxis], joint_best_ids_all]

        else:
            local_best_scores_all, local_best_ids_all = sess.run([score, ids],
                                                                 {topk_input: local_scores_all,
                                                           beam_size: beam})
        for k, hyp in enumerate(hyps):
            for j in range(beam):
                new_hyp = {}
                new_hyp['score'] = hyp['score'] + float(local_best_scores_all[k, j])
                new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                new_hyp['yseq'][len(hyp['yseq'])] = int(local_best_ids_all[k, j])
                if lpz is not None:
                    new_hyp['ctc_state_prev'] = ctc_states_all[k, joint_best_ids_all[k, j]]
                    new_hyp['ctc_score_prev'] = ctc_scores_all[k, joint_best_ids_all[k, j]]
                hyps_best_kept.append(new_hyp)
            hyps_best_kept = sorted(
                hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]
        hyps = hyps_best_kept

        if i == maxlen - 1:
            for hyp in hyps:
                if hyp['yseq'][-1] != eos:
                    hyp['yseq'].append(eos)

        remained_hyps = []
        for hyp in hyps:
            if hyp['yseq'][-1] == eos:
                if len(hyp['yseq']) > minlen:
                    hyp['score'] += (i+1) * penalty
                    ended_hyps.append(hyp)
            else:
                remained_hyps.append(hyp)

        if end_detect(ended_hyps, i):
            break

        hyps = remained_hyps

        if len(hyps) <= 0:
            break

    ended_hyps=sorted(
        ended_hyps, key=lambda x: x['score'], reverse=True)
    lenth = len(ended_hyps)
    if lenth == 0:
        return []
    index_l = [0]
    index_r = [len(ended_hyps)]
    for i in range(1,lenth):
        diff = len(ended_hyps[i]['yseq'])-len(ended_hyps[i-1]['yseq'])
        # 相邻分值差不多的 应该字数差不多  干掉字少的
        if abs(diff) > 3:
            if diff < 0:
                index_r.append(i)
            else:
                index_l.append(i)
        # 相邻分值相差很大 干掉前面分值大的 越大字越少
        elif ended_hyps[i]['score'] - ended_hyps[i - 1]['score'] < -10:
            index_r.append(i)
    m_l = max(index_l)
    m_r = min(index_r)

    if m_l >= m_r:
        m_r = lenth
    ended_hyps = ended_hyps[m_l:m_r]

    print('all scores: ', '\n')
    a_s = [i['score'] for i in ended_hyps]
    a_seq = [[han_vocab[b] for b in i['yseq']] for i in ended_hyps]
    for a, b in zip(a_s, a_seq):
        print(a, b)

    nbest_hyps = sorted(
        ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), nbest)]

    if len(nbest_hyps) == 0:
        return []

    result = [han_vocab[i] for i in nbest_hyps[0]['yseq'][1:-1]]
    return result


if __name__ == '__main__':
    # ctc权重  训练参数
    ctc_weight = cfg.PREDICT.CTC_WEIGHT

    CTC_SCORING_RATIO = cfg.PREDICT.CTC_SCORING_RATIO
    beam = cfg.PREDICT.BEAM
    penalty = cfg.PREDICT.PENALTY
    nbest = cfg.PREDICT.NBEST
    han_vocab = readtxt(cfg.BASE.DICT)
    # var_list = checkpoint_utils.list_variables(ckpt_file)
    # for v in var_list:
    #     print(v)
    graph_e2e = tf.Graph()

    #端到段模型初始化
    with graph_e2e.as_default():
        return_elements = ["input_data:0", "label_in:0", 'wave_l:0',
                         'encode_samples/enc_num_blocks_5/multihead_attention_1/ln/add_1:0',#encode节点名称
                         'LogSoftmax:0',  #encode_log_softmax 节点名称
                         'strided_slice:0'] #decode_log_softmax 节点名称

        pb_file = cfg.PREDICT.pb

        return_tensors = read_pb_return_tensors(graph_e2e, pb_file, return_elements)

        beam_size = tf.placeholder(tf.int32, shape=(), name='beam_size')
        topk_input= tf.placeholder(tf.float32, shape=(None, None), name='topk_input')
        #tensorflow 求topk
        score,ids = tf.nn.top_k(topk_input,k=beam_size)
        sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True),graph=graph_e2e)


    full_path = 'D4_750.wav'
    (data_input, sampling_rate) = soundfile.read(full_path)
    data_input = np.append(data_input[0], data_input[1:] - 0.95 * data_input[:-1])
    x = np.array(data_input)[:, np.newaxis]
    result = beam_search_decode_with_ctc(x, han_vocab, sess)
    result = ''.join(result)
    result = result.replace('$', ' ')
    print('result:', result)






