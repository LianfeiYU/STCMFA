
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

config = {
    "font.family": 'sans-serif',
    "font.size": 30,
    "mathtext.fontset": 'stix',
    "font.sans-serif": prop.get_name(),
}
rcParams.update(config)

i = 2

f = h5py.File('F:\\08_heads4_result.h5', 'r')
target = np.array(f['target'][i, :, 0]).reshape(-1)
predict = np.array(f['predict'][i, :, 0]).reshape(-1)

colors = np.random.rand(1000)  # 每个点对应的颜色值（介于0到1之间）

# 绘制散点图
fig = plt.figure(figsize=(12.5, 10), dpi=200)

plt.scatter(predict[:1000], target[:1000], c=colors, cmap='viridis', vmax=1, vmin=0)  # 'c'参数指定了颜色，'cmap'参数指定了颜色映射

# 添加颜色条
# colorbar_axes = plt.gca().add_artist(plt.colorbar())
plt.colorbar()

plt.ylabel("True data", size=45)
plt.xlabel("Predicted data", size=45)
plt.title("STCMFA on PeMS08", size=45)

plt.savefig('F:\\STCMFA_08.jpg', bbox_inches='tight')

# 显示图形
# plt.show()