import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import warnings
import datetime

warnings.filterwarnings('ignore')


features = pd.read_csv(r'C:\Users\Administrator\Desktop\pytorch学习\神经网络实战分类与回归任务\temps.csv')
#处理时间数据
years = features['year']
months = features['month']
days = features['day']

#datetime格式
dates = [str(int(years))+ '-'+str(int(months))+'-'+str(int(days))for years,months,days in zip(years,months,days)]

plt.style.use('fivethirtyeight')

# 准备画图
# 指定默认风格
plt.style.use('fivethirtyeight')

# 设置布局
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize = (10,10))
fig.autofmt_xdate(rotation = 45)

# 标签值
ax1.plot(dates, features['actual'])
ax1.set_xlabel(''); ax1.set_ylabel('Temperature'); ax1.set_title('Max Temp')

# 昨天
ax2.plot(dates, features['temp_1'])
ax2.set_xlabel(''); ax2.set_ylabel('Temperature'); ax2.set_title('Previous Max Temp')

# 前天
ax3.plot(dates, features['temp_2'])
ax3.set_xlabel('Date'); ax3.set_ylabel('Temperature'); ax3.set_title('Two Days Prior Max Temp')

# 我的逗逼朋友
ax4.plot(dates, features['friend'])
ax4.set_xlabel('Date'); ax4.set_ylabel('Temperature'); ax4.set_title('Friend Estimate')

plt.tight_layout(pad=2)

###学习掌握如何使用Python画图。
features = pd.get_dummies(features)
#独热编码生成虚拟变量dummis
lables = np.array(features['actual'])
features = features.drop('actual',axis=1)

features_list = list(features.columns)
features= np.array(features)

from sklearn import preprocessing
input_features = preprocessing.StandardScaler().fit_transform(features)

x = torch.tensor(input_features,dtype=float)
y = torch.tensor(lables,dtype=float)

#定义模型
weights = torch.randn((14, 128), dtype = float, requires_grad = True)
biases = torch.randn(128, dtype = float, requires_grad = True)
weights2 = torch.randn((128, 1), dtype = float, requires_grad = True)
biases2 = torch.randn(1, dtype = float, requires_grad = True)

learning_rate = 0.001
losses =[]

for i in range(1000):
    hidden =x.mm(weights) + biases
    hidden = torch.relu(hidden)
    predictions = hidden.mm(weights2) + biases2
    loss=torch.mean((predictions - y)**2)
    losses.append(loss.item())

    if i%100 ==0:
        print(loss)
    loss.backward()

    weights.data.add_(- learning_rate * weights.grad.data)
    biases.data.add_(- learning_rate * biases.grad.data)
    weights2.data.add_(- learning_rate * weights2.grad.data)
    biases2.data.add_(- learning_rate * biases2.grad.data)

    weights.grad.data.zero_()
    biases.grad.data.zero_()
    weights2.grad.data.zero_()
    biases2.grad.data.zero_()

print(predictions[:10])