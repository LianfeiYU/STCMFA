import pandas as pd
import numpy as np
import h5py

import warnings
warnings.filterwarnings('ignore')

#matplotlib相关
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import font_manager
from pylab import *
font_path = "E:\\tnw+simsun.ttf"
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
from matplotlib import rcParams #内部具体设置字体大小

f = h5py.File('F:\\08_heads4_result.h5', 'r')

target = f['target'][:, 22:, :]
predict = f['predict'][:, 22:, :]

f2 = h5py.File('F:\\STGODE_result.h5', 'r')

target2 = f2['target'][:]
predict2 = f2['predict'][:]
print(target.shape)
print(predict.shape)
print(target2.shape)
print(predict2.shape)

# PeMS08数据集可视化

config = {
    "font.family": 'sans-serif',
    "font.size": 25,
    "mathtext.fontset": 'stix',
    "font.sans-serif": prop.get_name(),
}
rcParams.update(config)

for i in range(0, 170):  # target.shape[0]
    co = [-288, None]

    Target = target[i, :-12, 11].reshape(-1).tolist()
    STC_5 = predict[i, 12:, 0].reshape(-1).tolist()
    STC_60 = predict[i, :-12, 11].reshape(-1).tolist()
    STG_5 = predict2[i, 12:, 0].reshape(-1).tolist()
    STG_60 = predict2[i, :-12, 11].reshape(-1).tolist()


    plt.figure(figsize=(14, 10), dpi=450)  # figsize=(30,8),

    xs = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00", ]
    scale = np.arange(0, 289, 48)  # 设置每1920个横坐标刻度显示一次

    plt.xticks(scale, xs)  # 自己定义横坐标
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(48))  # 设置每1920个横坐标刻度显示一次

    plt.plot(STC_5[co[0]:co[1]],marker='d',linewidth=1, color='r',label="STCMFA-5mins") #,linewidth=1
    plt.plot(STC_60[co[0]:co[1]],marker='x',linewidth=1, color='m',label="STCMFA-60mins") #,linewidth=1
    plt.plot(STG_5[co[0]:co[1]],marker='p',linewidth=1, color='b',label="STGODE-5mins") #,linewidth=1
    plt.plot(STG_60[co[0]:co[1]],marker='*',linewidth=1, color='c',label="STGODE-60mins") #,linewidth=1
    plt.plot(Target[co[0]:co[1]],marker='o',linewidth=1, color='black',label="Ground Truth") #,linewidth=1

    plt.legend(loc='best')

    plt.xlabel("Time steps (5 mins)", size=32)
    plt.ylabel("Traffic flow", size=32)
    plt.title("Node #{0} in PeMS08".format(i+1), size=32)

    plt.savefig('F:\\08_{0}.jpg'.format(i+1), bbox_inches = 'tight')

#     plt.show()
