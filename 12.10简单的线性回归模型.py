import torch
import torch.nn as nn
import numpy as np
import os

# 定义输入数据的值
x_values = [i for i in range(11)]
# 将输入值转换为浮点数的numpy数组，并调整其形状
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

# 定义目标输出的值
y_values = [2*i + 1 for i in x_values]
# 将输出值转换为浮点数的numpy数组，并调整其形状
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)
# 定义线性回归模型类
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        # 调用父类初始化方法
        super(LinearRegressionModel, self).__init__()
        # 定义线性层
        self.linear = torch.nn.Linear(input_dim, output_dim)

    # 定义前向传播方法
    def forward(self, x):
        out = self.linear(x)
        return out

# 输入和输出维度
input_dim = 1
output_dim = 1

# 实例化线性回归模型
model = LinearRegressionModel(input_dim, output_dim)

#这段代码的作用是设置 PyTorch 的设备。如果当前系统中有可用的 GPU（通过判断`torch.cuda.is_available()`是否为真），则将设备设置为第一个 GPU（"cuda:0"）；如果没有可用的 GPU，则将设备设置为 CPU。这样可以根据系统的硬件条件自动选择合适的计算设备，以提高计算效率。
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
# 定义训练参数
epochs = 1000

learning_rate = 0.01
# 定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# 定义损失函数
#model.parameters()：
#在 PyTorch 中，model通常是一个继承自torch.nn.Module的神经网络模型对象。model.parameters()是一个生成器函数，它返回模型中所有需要被学习（即可以被更新）的参数。这些参数包括神经网络中的各种权重（例如线性层中的权重矩阵）和偏置（例如线性层中的偏置向量）。对于一个复杂的多层神经网络，model.parameters()会遍历模型的所有层，收集所有可学习的参数，并将它们作为一个整体提供给优化器。这样，优化器就知道哪些参数需要根据损失函数的梯度进行调整。
#lr = learning_rate：
#lr是学习率（learning rate）的缩写，它是优化器的一个重要超参数。学习率决定了每次参数更新的步长大小。learning_rate是一个预先定义好的数值（例如可以是 0.001、0.01 等），它控制着优化器在更新模型参数时的速度。如果学习率设置得过大，可能会导致优化器在参数空间中 “跳跃” 过大，错过损失函数的最小值，甚至可能导致损失函数的值无法收敛，反而发散；如果学习率设置得过小，参数更新会非常缓慢，导致训练过程需要很长时间才能达到较好的拟合效果。

criterion = torch.nn.MSELoss()

# 开始模型训练
for epoch in range(epochs):
    epoch += 1
    inputs = torch.from_numpy(x_train)
    inputs = inputs.to(device)
    labels = torch.from_numpy(y_train)
    labels = labels.to(device)
    # 在 PyTorch 中，在每次进行反向传播计算梯度并更新模型参数之前，需要先将之前累积的梯度清零。因为在默认情况下，梯度是会累积的（例如在处理循环神经网络等场景，或者多次调用 backward() 方法时，如果不清零，梯度就会不断叠加）。optimizer.zero_grad() 这个方法调用就是让优化器 optimizer（前面应该已经创建好了，比如是 SGD、Adam 等类型的优化器）将其管理的模型所有参数对应的梯度清零，为本次迭代中即将进行的前向传播、计算损失、反向传播以及后续的参数更新操作做好准备，确保梯度计算和参数更新是基于当前这一轮次（epoch）的数据和模型状态进行的，避免受到之前轮次残留梯度的影响。
    optimizer.zero_grad()
    # 获取模型输出
    outputs = model(inputs)
    outputs = outputs.to(device)
    #(要把张量和线性层都移动到同一个设备上，才能进行后续的计算)
    # 计算损失
    loss = criterion(outputs, labels)
    # 反向传播:这是基于计算得到的损失值 loss 进行反向传播操作。反向传播算法依据链式法则，自动计算损失值关于模型中各个可学习参数（比如各层的权重和偏置）的梯度。通过这一步骤，模型就知道了每个参数应该朝着哪个方向调整（根据梯度的正负）以及调整的幅度大致与梯度大小相关，为后续的参数更新提供依据。在调用 loss.backward() 之后，模型各参数对应的梯度就被计算并存储好了，等待下一步的参数更新操作。
    loss.backward()
    # 更新模型参数:调用优化器 optimizer（例如前面创建的 torch.optim.SGD 等类型的优化器）的 step 方法，它会根据之前反向传播计算得到的各参数的梯度，结合优化器自身设定的参数更新策略（不同优化器有不同的更新规则，像随机梯度下降 SGD 按照一定学习率和梯度方向更新参数等）以及设定的学习率等超参数，对模型的各可学习参数进行更新。更新后的模型参数就会应用到下一轮的训练中，随着一轮轮的训练不断迭代，模型就逐渐学习到数据中的模式，使预测结果更贴合真实标签，达到优化模型性能的目的。
    optimizer.step()

    # 每100个epoch打印一次损失值
    if epoch % 100 == 0:
        print(f"epoch: {epoch}, loss: {loss.item():.32f}")

 #这行代码的作用是使用 PyTorch 框架将模型的状态字典（包含模型的参数）保存到名为'model.pkl'的文件中。这样可以在以后的时间里重新加载这个状态字典来恢复模型的参数，以便继续训练或进行预测等操作。
 #@torch.save(model.state_dict(), 'model.pkl')#默认保存到当前目录下C:\Users\Administrator\PycharmProjects\pythonProject
 #@torch.save(model.state_dict(), 'C:\Users\Administrator\PycharmProjects\pythonProject\model.pkl')#保存到指定目录下
 #这段代码的作用是使用 PyTorch 框架加载已保存的模型参数。其中，“model”是一个已经定义好的模型对象，“load_state_dict”是用于加载模型参数状态字典的方法，“torch.load('model.pkl')”表示从名为“model.pkl”的文件中加载保存的模型参数状态字典。这样可以恢复之前训练好的模型状态，以便进行进一步的预测、评估或继续训练等操作。
#model.load_state_dict(torch.load('model.pkl'))
print(os.getcwd())