# speech-regconize
语音识别

1 环境安装
   pip install soundfile
   pip install tensorflow-gpu==1.12

2 运行
    python decoder.py

3 训练

数据准备：
	见data文件夹 txt格式 音频路径+'\t' + label (label用红歌分割)
	config.py 中data_path+音频路径  为音频的绝对路径
python generate_data.py 不保错 则数据准备正确
运行 python train.py 训练 
