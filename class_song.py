#-*- conding:utf-8 -*-
import pandas as pd
import numpy as np
from pydub.audio_segment import AudioSegment#pydub是处理音乐文件的一个库
from scipy.io import wavfile
from python_speech_features import mfcc
import os
import sys
song =AudioSegment.from_file('./我们的纪念.mp3',format="mp3")#先读一下
#切分歌曲
# song_split=song[-30*100:]
song.export("./我们的纪念.wav",format='wav')#转化为wav格式
rate,data=wavfile.read("./我们的纪念.wav")#每秒读取速度以及数据
print(rate)
print(data.shape)
mf_feat = mfcc(data,rate,numcep=13,nfft=2048)#数据转化为13维，频率改为2048
print(mf_feat.shape)
mm =np.mean(mf_feat,axis=0)#隐含了时序上的相关性
print(mm.shape)
mf = np.transpose(mf_feat)
mc = np.cov(mf)#协方差矩阵，里面的值也就是各个特征之间的协方差
print(mc.shape)
result = mm
#np.diag(mc,k)方针里面的值，k=0为对角线的元素，
for k in range(len(mm)):
    result=np.append(result,np.diag(mc,k))
print(result.shape)


