import pandas as pd
import numpy as np
import h5py

import warnings
warnings.filterwarnings('ignore')

from matplotlib import font_manager
from pylab import *
font_path = "E:\\tnw+simsun.ttf"
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
from matplotlib import rcParams #内部具体设置字体大小


f = h5py.File('F:\\08_heads4_result.h5', 'r')
target = f['target'][:]
predict = f['predict'][:]

abs_loss = abs(predict-target)

last = []
k = 0
for i in range(0, abs_loss.shape[0]):
    no = abs_loss[i, :61, 11]
    last.append(no)
    k = k + 1
    if k==12:
        break
last = np.array(last)


from matplotlib import cm
import seaborn as sns

config = {
    "font.family":'sans-serif',
    "font.size": 12,
    "mathtext.fontset":'stix',
    "font.sans-serif": prop.get_name(),
}
rcParams.update(config)


plt.figure(figsize=(16, 4),dpi=350) #figsize=(16,10),

scale =[i for i in range(3,61,4)]#坐标
xs = [i for i in range(4,61,4)] #坐标显示值
# plt.xticks(scale, xs)   #自己定义横坐标
# plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(4))
yt = [i for i in range(1,13)]
yt.reverse()
# print(yt)


# plt.imshow(test, cmap='autumn_r') #plasma_r,YlOrRd
tu = sns.heatmap(last, cmap='cool',xticklabels = 4, yticklabels = yt, cbar_kws={"pad": 0.01, "shrink": 0.675}, vmax=25, vmin=0, linewidths=0.6, square = True) #, cbar_kws={"shrink": 0.614}
# tu.set_xticks(scale)

# plt.colorbar()
# plt.tight_layout()
plt.xlabel("Time steps (5 mins)",size=30)
plt.ylabel("Roads",size=30)
plt.title("PeMS08 (60 mins)",size=30)

plt.savefig('E:\\08_60mins.jpg',bbox_inches = 'tight')

plt.show()