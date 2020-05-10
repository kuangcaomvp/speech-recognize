#! /usr/bin/env python
# coding=utf-8
from easydict import EasyDict as edict

__C                             = edict()
cfg                             = __C

# BASE options
__C.BASE                        = edict()

#数据字典
__C.BASE.DICT                  = 'config/key.txt'


#Predict options

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


