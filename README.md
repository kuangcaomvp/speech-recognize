# speech-regconize
语音识别

1 环境安装

       pip install soundfile
       
       pip install tensorflow-gpu==1.12
       
       pip install python_speech_features
       
       pip install tqdm
       
       pip install easydict
       
       cuda9.0

2 测试

        python decoder.py

3 训练

        数据准备：
        
            见data文件夹 txt格式 音频路径+'\t' + label (label用空格分割)
        
            config.py 中data_path+音频路径  为音频的绝对路径
        
            运行 python generate_data.py 不报错 则数据准备正确
        
        运行 python train.py 进行训练 

4 模型冻结 
    
        freeze_graph.py  修改ckpt_file为自己训练的checkpoint路径
                          
                         pb_file   生成的pb文件保存路径
        
        运行  python freeze_graph.py
        
        修改 config.py 中__C.PREDICT.pb = pb_file
    
    运行python decode.py 测试

