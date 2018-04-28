#coding:utf-8
import pandas as pd
import numpy as np
import glob#搜寻文件的包
from pydub.audio_segment import AudioSegment
from scipy.io import wavfile#把mp3格式转化为wav格式
from python_speech_features import mfcc#傅里叶变幻和梅尔倒谱系数处理声音的
import os
import sys
import time



def 获取歌单(file):#extract_label
    data = pd.read_csv(file)
    data = data[['name','tag']]
    return data
def 获取单首歌曲特征(file):#fetch_index_label
    items = file.split('.')
    file_format = items[-1].lower()#获取歌曲格式
    file_name = file[: -(len(file_format)+1)]#获取歌曲名称
    if file_format != 'wav':
        song = AudioSegment.from_file(file, format = 'mp3')
        file = file_name + '.wav'
        song.export(file, format = 'wav')
    try:
        rate, data = wavfile.read(file)
        mfcc_feas = mfcc(data, rate, numcep = 13, nfft = 2048)
        mm = np.transpose(mfcc_feas)#先LDA再降唯，如果不进行转置？
        mf = np.mean(mm ,axis = 1)# mf变成104维的向量
        mc = np.cov(mm)
        result = mf
        for i in range(mm.shape[0]):
            result = np.append(result, np.diag(mc, i))
#         os.remove(file)
        return result
    except Exception as msg:
        print(msg)
# 获取单首歌曲特征("我们的纪念.mp3")
# sys.exit("0")
def 特征提取主函数():#主函数extract_and_export
    df = 获取歌单("./data/music_info.csv")
    name_label_list = np.array(df).tolist()
    name_label_dict = dict(map(lambda t: (t[0], t[1]), name_label_list))#歌单做成字典
    labels = set(name_label_dict.values())
    print(labels)
    label_index_dict = dict(zip(labels, np.arange(len(labels))))#歌曲标签数值映射
#    print(label_index_dict)
#    for k in label_index_dict:
#        print(k)
#        print(label_index_dict[k])
#    sys.exit(0)

    all_music_files = glob.glob("./data/music/*.mp3")
    all_music_files.sort()
     
    loop_count = 0
    flag = True
     
    all_mfcc = np.array([])
    for file_name in all_music_files:
        print('开始处理：' + file_name.replace('\xa0', ''))#.replace('\xa0', '')
        music_name = file_name.split('\\')[-1].split('.')[-2].split('-')[-1]#\为转意字符
        music_name = music_name.strip()
        if music_name in name_label_dict:
            label_index = label_index_dict[name_label_dict[music_name]]
            ff = 获取单首歌曲特征(file_name)
            ff = np.append(ff, label_index)
             
            if flag:
                all_mfcc = ff
                flag = False
            else:
                all_mfcc = np.vstack([all_mfcc, ff])
        else:
            print('无法处理：' + file_name.replace('\xa0', '') +'; 原因是：找不到对应的label')
        print('looping-----%d' % loop_count)
        print('all_mfcc.shape:', end='')
        print(all_mfcc.shape)
        loop_count +=1
    #保存数据
    label_index_list = []
    for k in label_index_dict:
        label_index_list.append([k, label_index_dict[k]])
    pd.DataFrame(label_index_list).to_csv(数值化标签路径, header = None, \
                                          index = False, encoding = 'utf-8')
    pd.DataFrame(all_mfcc).to_csv(歌曲特征文件存放路径, header= None, \
                                  index =False, encoding='utf-8')
    return all_mfcc

if __name__=='__main__':
    歌单路径 = './data/music_info.csv'#music_info_csv_file_path
    歌曲源路径 = './data/music/*.mp3'#music_audio_dir
    数值化标签路径 = './data/music_index_label.csv'#music_index_label_path
    歌曲特征文件存放路径 = './data/music_features.csv'#music_features_file_path
    start = time.time()
    特征提取主函数()
    end = time.time()
    print('总耗时%.2f秒'%(end - start))
        
       
    
    
    
    
    
    
    
    