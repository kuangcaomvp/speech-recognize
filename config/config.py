#! /usr/bin/env python
# coding=utf-8
from easydict import EasyDict as edict

__C                             = edict()
cfg                             = __C

# BASE options
__C.BASE                        = edict()

#数据字典
__C.BASE.DICT                  = 'config/key.txt'

__C.BASE.DATA_PATH              = '/e'

# 使用mel fbank或者mfcc作特征提取
__C.BASE.FEATURE_TYPE            = 'mel'   #['mel','mfcc']
__C.BASE.OUTPUT_NUM            = {'mel': 40,'mfcc': 26}
#是否均值化
__C.BASE.NOMALNIZE             = 'none'   #['none', 'local', 'local_scalar','cmvn']
__C.BASE.initial_weight        = 'logs_lm_tf/lm_train_loss=2.8582.ckpt-2'
#Predict options

#迭代
__C.BASE.epochs                =30
__C.BASE.warmup_periods        =10
#学习率
__C.BASE.lr                    =1e-4
__C.BASE.lr_deep               =1e-6

# Train options
__C.TRAIN                       = edict()

__C.TRAIN.BATCH_SIZE            = 12
__C.TRAIN.read_files            = ['aishell_train.txt']

# TEST options
__C.TEST                        = edict()
__C.TEST.BATCH_SIZE            = 12
__C.TEST.read_files             =['aishell_dev.txt']

__C.PREDICT                    = edict()

# 训练是ctc占比权重  用于解码
__C.PREDICT.CTC_WEIGHT         = 0.3

#ctc beam链路的占比
__C.PREDICT.CTC_SCORING_RATIO  = 1.5

#beam size
__C.PREDICT.BEAM               = 10

__C.PREDICT.PENALTY            = 0.0

# 最终去最好的几个结果
__C.PREDICT.NBEST              = 1

# pb 模型文件的地址 端到段模型
__C.PREDICT.pb                  = 'pb_save/model.pb'


