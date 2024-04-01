
import warnings
warnings.filterwarnings('ignore')

#matplotlib相关
import matplotlib.pyplot as plt
import skill_metrics as sm

import matplotlib.ticker as ticker
from matplotlib import font_manager
from pylab import *
font_path = "E:\\tnw+simsun.ttf"
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
from matplotlib import rcParams #内部具体设置字体大小

config = {
    "font.family":'sans-serif',
    "font.size": 11,
    "mathtext.fontset":'stix',
    "font.sans-serif": prop.get_name(),
}
rcParams.update(config)

# PeMS08
sd = np.array([])  # 计算后输入
cc = np.array([])  # 计算后输入
rmsd = np.array([0,0,0,0,0])

label=['obs','LSTM','DCRNN','STGODE','STCMFA']
cor = ['k','k','r','r','b']

fig = plt.figure(figsize=(12,8),dpi=250)
ax = fig.add_axes([0.1, 0.6, 0.6, 0.6])
# 绘图核心函数
sm.taylor_diagram(sd,rmsd,cc,markerLabel = label, markerLegend = 'on', markerSize=6,#markercolor="g", markerSize=7, #markerLegend = 'on',#基本参数
                  colCOR="mediumslateblue",styleCOR="--",widthCOR=0.8, #CC相关设置
                  colSTD="k",widthSTD=0.8,styleSTD="--" ,axismax=200, #SD相关设置
                  widthRMS=0.9,labelRMS='RMSD',colRMS='c', showlabelsRMS = 'off',#RMSD相关设置
                  colOBS="orangered",styleOBS="-",widthOBS=1.3,markerObs="D",titleOBS="Observation", #观测值设置
                 )

plt.title("        PeMS08",size=17)
ax.grid(False)

plt.savefig('F:\\泰勒图08.jpg',bbox_inches='tight')
plt.show()