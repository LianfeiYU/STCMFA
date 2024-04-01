
import warnings
warnings.filterwarnings('ignore')

from matplotlib import font_manager
from pylab import *
font_path = "E:\\tnw+simsun.ttf"
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
from matplotlib import rcParams #内部具体设置字体大小

f = open('F:\\training\\1.txt', encoding='gb18030')
lines03 = f.readlines()
li03 = []
for line in lines03:
    if line.find("Eval Loss: ") != (-1):
        # print(line.split(" ")[-1])
        numbers = re.findall(r'\d+\.\d+', line)
        li03.append(float(numbers[1]))

f = open('F:\\training\\2.txt', encoding='gb18030')
lines04 = f.readlines()
li04 = []
for line in lines04:
    if line.find("Eval Loss: ") != (-1):
        # print(line.split(" ")[-1])
        numbers = re.findall(r'\d+\.\d+', line)
        li04.append(float(numbers[1]))

f = open('F:\\training\\3.txt', encoding='gb18030')
lines07 = f.readlines()
li07 = []
for line in lines07:
    if line.find("Eval Loss: ") != (-1):
        # print(line.split(" ")[-1])
        numbers = re.findall(r'\d+\.\d+', line)
        li07.append(float(numbers[1]))

f = open('F:\\training\\4.txt', encoding='gb18030')
lines08 = f.readlines()
li08 = []
for line in lines08:
    if line.find("Eval Loss: ") != (-1):
        # print(line.split(" ")[-1])
        numbers = re.findall(r'\d+\.\d+', line)
        li08.append(float(numbers[1]))


print(len(li03))
print(len(li04))
print(len(li07))
print(len(li08))

#Data-01数据集可视化

config = {
    "font.family":'sans-serif',
    "font.size": 23,
    "mathtext.fontset":'stix',
    "font.sans-serif": prop.get_name(),
}
rcParams.update(config)

plt.figure(figsize=(21,8),dpi=350) #figsize=(30,8),

plt.plot(li03[:200], linewidth=2, color='r',label="PeMS03") #, marker='*', linewidth=1
plt.plot(li04[:200], linewidth=2, color='orange',label="PeMS04") #, marker='*', linewidth=1
plt.plot(li07[:200], linewidth=2, color='limegreen',label="PeMS07") #, marker='*', linewidth=1
plt.plot(li08[:200], linewidth=2, color='slateblue',label="PeMS08") #, marker='*', linewidth=1

plt.legend(loc='best')

plt.xlabel("Epochs")
plt.ylabel("Loss")
# plt.title("Node #{0} in PeMS08",size=20)

plt.savefig('F:\\train_loss.jpg',bbox_inches = 'tight')

plt.show()