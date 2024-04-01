import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler

#matplotlib相关
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import font_manager
from pylab import *
font_path = "E:\\tnw+simsun.ttf"
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
from matplotlib import rcParams #内部具体设置字体大小

df_RMSE = pd.read_csv(r"F:\\RMSE_08.csv")

config = {
    "font.family":'sans-serif',
    "font.size": 20,
    "mathtext.fontset":'stix',
    "font.sans-serif": prop.get_name(),
}
rcParams.update(config)

scale = range(0,12) #坐标
xs = ["5","10","15","20","25","30","35","40","45","50","55","60"] #坐标显示值

plt.figure(figsize=(9,6),dpi=400) #figsize=(16,10),

plt.xticks(scale, xs)   #自己定义横坐标
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))  #设置每1920个横坐标刻度显示一次

ylim(19, 31)


plt.plot(df_RMSE['ch1'][:12].values,marker='o', color='darkorange',label="Only-Spatial",linewidth=3)
plt.plot(df_RMSE['ch3'][:12].values,marker='o', color='c',label="REPL-FAtt",linewidth=3)
plt.plot(df_RMSE['ch4'][:12].values,marker='o', color='hotpink',label="RM-TS",linewidth=3)
plt.plot(df_RMSE['ch2'][:12].values,marker='o', color='yellowgreen',label="REPL-Agg",linewidth=3)
plt.plot(df_RMSE['ch5'][:12].values,marker='o', color='orangered',label="STCMFA",linewidth=3)

plt.ylabel("PeMS08 RMSE")
plt.xlabel("Prediction Horizon (5 mins)")

plt.legend(loc='best')

plt.savefig('F:\\RMSE_08.jpg', bbox_inches='tight')

plt.show()

