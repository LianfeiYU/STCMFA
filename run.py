
# @Time    : 2024/04/01
# @Author  : Lianfei Yu
# @github  : https://github.com/

import os
import time
from datetime import datetime
import h5py
import torch
import numpy as np
import random

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader import LoadData

from utils import Evaluation, masked_mae_np
from utils import visualize_result

from early_stopping import EarlyStopping

from stcmfa import STCMFA

import warnings
warnings.filterwarnings('ignore')


def main(no, i):
    all_start_time = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # PEMS03: num_nodes=358, total_days=91天   26208条(2018.09.01-2018.11.30),时间粒度:5min
    # PEMS04: num_nodes=307, total_days=59天   16992条(2018.01.01-2018.02.28),时间粒度:5min
    # PEMS07: num_nodes=883, total_days=98天   28224条(2017.05.01-2017.08.31),时间粒度:5min
    # PEMS08: num_nodes=170, total_days=62天   17856条(2016.07.01-2016.08.31),时间粒度:5min

    # 训练多批数据
    data_no = no
    jj = i

    bs = 16

    data_path = []

    data_path.append("PeMS_{0}/PeMS{1}.csv".format(data_no, data_no))
    data_path.append("PeMS_{0}/PeMS{1}.npz".format(data_no, data_no))

    if data_no == "03":
        num_nodes = 358
        total_days = 91
    elif data_no == "04":
        num_nodes = 307
        total_days = 59
    elif data_no == "07":
        num_nodes = 883
        total_days = 98
    elif data_no == "08":
        num_nodes = 170
        total_days = 62


    # 第一步：准备数据（上一节已经准备好了，这里只是调用而已，链接在最开头）
    train_data = LoadData(data_path=[data_path[0], data_path[1]], num_nodes=num_nodes, total_days=total_days,
                          time_interval=5, history_length=12,
                          train_mode="train")

    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=0)  # num_workers是加载数据（batch）的线程数目


    validate_data = LoadData(data_path=[data_path[0], data_path[1]], num_nodes=num_nodes, total_days=total_days,
                         time_interval=5, history_length=12,
                         train_mode="validate")
    validate_loader = DataLoader(validate_data, batch_size=bs, shuffle=False, num_workers=0)


    test_data = LoadData(data_path=[data_path[0], data_path[1]], num_nodes=num_nodes, total_days=total_days,
                         time_interval=5, history_length=12,
                         train_mode="test")
    test_loader = DataLoader(test_data, batch_size=bs, shuffle=False, num_workers=0)#32


    my_net = STCMFA(in_c=jj, hid_c=256, out_c=64, K=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 定义设备

    my_net = my_net.to(device)  # 模型送入设备

    # 打印日志
    current_time = datetime.now().strftime('%Y-%m-%d %H%M %S')
    current_dir = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(current_dir, 'experiments', "PeMS" + data_no, current_time)
    if os.path.isdir(log_dir) == False and not False:  # debug True
        os.makedirs(log_dir, exist_ok=True)
    # print(log_dir)

    # 第三步：定义损失函数和优化器
    criterion = torch.nn.L1Loss().to(device)

    optimizer = optim.Adam(params=my_net.parameters(), lr=1e-3)#, lr=1e-4    默认学习率为lr=1e-3

    # 第四步：训练+测试
    # Train model
    Epoch = 200  # 训练的次数

    train_losses = []# 用数组保存每一轮迭代中，在测试数据上测试的损失值和精确度，也是为了通过画图展示出来。
    eval_losses = []

    save_path = log_dir+"\\"+"{0}_heads{1}_best_model.pt".format(data_no, jj)

    patience = 40
    early_stopping = EarlyStopping(save_path, patience)

    # ================================================================================================
    # ================================================================================================
    # 01 模型训练、验证

    for epoch in range(1, Epoch + 1):

        # == == == == == == == == == == == == == ① 训练模式 == == == == == == == == == == == == ==
        epoch_loss = 0.0
        count = 0
        start_time = time.time()

        my_net.train()  # 打开训练模式

        for data in train_loader:  # ["graph": [B, N, N] , "flow_x": [B, N, H, D], "flow_y": [B, N, 1, D]],一次把一个batch的训练数据取出来
            my_net.zero_grad()  # 梯度清零
            count +=1

            predict_value = my_net(data, device).to(torch.device("cpu"))  # [B, N, 1, D],由于标签flow_y在cpu中，所以最后的预测值要放回到cpu中

            # predict_value = my_net(data, device)[0].to(
            #     torch.device("cpu"))  # opt_lstm专用

            # print(predict_value.size())
            # print(data["flow_x"].size())
            # print(data["flow_y"].size())
            loss = criterion(predict_value, data["flow_y"])  # 计算损失，切记这个loss不是标量

            epoch_loss += loss.item()  # 这里是把一个epoch的损失都加起来，最后再除训练数据长度，用平均loss来表示

            loss.backward()  # 反向传播

            optimizer.step()  # 更新参数

        train_losses.append(epoch_loss / len(train_data))#每轮的存到数组中，以备后序画图


        # == == == == == == == == == == == == == ② 验证模式 == == == == == == == == == == == == ==
        eval_loss = 0

        my_net.eval()  # 将模型改为预测模式

        with torch.no_grad():
            for data in validate_loader:
                predict_value = my_net(data, device).to(torch.device("cpu"))

                loss = criterion(predict_value, data["flow_y"])

                eval_loss += loss.item()

            eval_losses.append(eval_loss / len(validate_data))

            end_time = time.time()

            print("Epoch: {:03d}     Train Loss: {:02.4f}     Eval Loss: {:02.4f}     Time: {:02.2f} mins".format(epoch,1000 * epoch_loss / len(
                                                                                                                      train_data),
                                                                                                                  1000 * eval_loss / len(
                                                                                                                      validate_data),
                                                                                                                  (
                                                                                                                              end_time - start_time) / 60))

            # 早停机制
            early_stopping(eval_loss, my_net)
            # 达到早停止条件时，early_stop会被置为True
            if early_stopping.early_stop:
                print("\nEarly stopping!!!")
                epoch = epoch - patience
                break  # 跳出迭代，结束训练

        # scheduler.step()  # 用于改变学习率

    #================================================================================================
    # ================================================================================================
    # 02 模型测试

    # 对于测试:
    # 第一、除了计算loss之外，还需要可视化一下预测的结果（定性分析）
    # 第二、对于预测的结果这里我使用了 MAE, MAPE, and RMSE 这三种评价标准来评估（定量分析）


    my_net.load_state_dict(torch.load(save_path))
    my_net.eval()  # 打开测试模式
    with torch.no_grad():  # 关闭梯度
        MAE, MAPE, RMSE = [], [], []# 定义三种指标的列表
        Target = np.zeros([num_nodes, 1, 12])  # [N, T, D],T=1 # 目标数据的维度，用０填充   #170
        Predict = np.zeros_like(Target)  # [N, T, D],T=1 # 预测数据的维度

        total_loss = 0.0
        for data in test_loader:  # 一次把一个batch的测试数据取出来

            # 下面得到的预测结果实际上是归一化的结果，有一个问题是我们这里使用的三种评价标准以及可视化结果要用的是逆归一化的数据
            predict_value = my_net(data, device).to(torch.device("cpu"))  # [B, N, 1, D]，B是batch_size, N是节点数量,1是时间T=1, D是节点的流量特征
            # print("predict_value1", predict_value.shape)
            loss = criterion(predict_value, data["flow_y"])  # 使用MSE计算loss

            total_loss += loss.item()  # 所有的batch的loss累加

            predict_value = predict_value.transpose(2, 3).transpose(0, 2).squeeze(0)  # [1, N, B(T), D] -> [N, B(T), D] -> [N, T, D]
            # print("predict_value2", predict_value.shape)
            target_value = data["flow_y"].transpose(2, 3).transpose(0, 2).squeeze(0)  # [1, N, B(T), D] -> [N, B(T), D] -> [N, T, D]

            # print(predict_value.shape)
            # print(target_value.shape)
            performance, data_to_save = compute_performance(predict_value, target_value, test_loader)  # 计算模型的性能，返回评价结果和恢复好的数据

            # 下面这个是每一个batch取出的数据，按batch这个维度进行串联，最后就得到了整个时间的数据，也就是
            # [N, T, D] = [N, T1+T2+..., D]
            # print(Predict.shape)
            # print(Target.shape)
            #
            # print(predict_value.shape)
            # print(target_value.shape)
            #
            # print(data_to_save[0].shape)
            # print(data_to_save[1].shape)

            Predict = np.concatenate([Predict, data_to_save[0]], axis=1)
            Target = np.concatenate([Target, data_to_save[1]], axis=1)

            MAE.append(performance[0])
            MAPE.append(performance[1])
            RMSE.append(performance[2])

        print("\nTest Loss: {:02.4f}".format(1000 * total_loss / len(test_data)))



    all_end_time = time.time()

    all_time = all_end_time - all_start_time
    hour = int(all_time/3600)
    min = int((all_time%3600)/60)

    if hour<1:
        all_time = str(min) + "mins"
    else:
        all_time = str(hour) + "h " + str(min) + "mins"


    # 三种指标取平均
    print("End_epoch:{}   MAE: {:2.4f}   MAPE: {:2.4f}   RMSE: {:2.4f}   Alltime: {}\n".format(epoch, np.mean(MAE), np.mean(MAPE), np.mean(RMSE), all_time))


    Predict = np.delete(Predict, 0, axis=1) # 将第0行的0删除，因为开始定义的时候用0填充，但是时间是从1开始的
    Target = np.delete(Target, 0, axis=1)

    result_file = log_dir+"\\"+"{0}_heads{1}_result.h5".format(data_no, jj)
    file_obj = h5py.File(result_file, "w")  # 将预测值和目标值保存到文件中，因为要多次可视化看看结果

    file_obj["predict"] = Predict  # [N, T, D]
    file_obj["target"] = Target  # [N, T, D]


def compute_performance(prediction, target, data):  # 计算模型性能
    # 下面的try和except实际上在做这样一件事：当训练+测试模型的时候，数据肯定是经过dataloader的，所以直接赋值就可以了
    # 但是如果将训练好的模型保存下来，然后测试，那么数据就没有经过dataloader，是dataloader型的，需要转换成dataset型。
    try:
        dataset = data.dataset  # 数据为dataloader型，通过它下面的属性.dataset类变成dataset型数据
    except:
        dataset = data  # 数据为dataset型，直接赋值

    # 下面就是对预测和目标数据进行逆归一化，recover_data()函数在上一小节的数据处理中
    #  flow_norm为归一化的基，flow_norm[0]为最大值，flow_norm[1]为最小值
    # prediction.numpy()和target.numpy()是需要逆归一化的数据，转换成numpy型是因为 recover_data()函数中的数据都是numpy型，保持一致
    prediction = LoadData.recover_data(dataset.flow_norm[0], dataset.flow_norm[1], prediction.numpy())
    target = LoadData.recover_data(dataset.flow_norm[0], dataset.flow_norm[1], target.numpy())


    # 对三种评价指标写了一个类，这个类封装在另一个文件中，在后面
    mae, mape, rmse = Evaluation.total(target.reshape(-1), prediction.reshape(-1))  # 变成常向量才能计算这三种指标

    # mae, mape, rmse = Evaluation.total(target.numpy().reshape(-1), prediction.numpy().reshape(-1))  # 变成常向量才能计算这三种指标

    performance = [mae, mape, rmse]
    recovered_data = [prediction, target]

    return performance, recovered_data  # 返回评价结果，以及恢复好的数据（为可视化准备的）


if __name__ == '__main__':
    # main()
    data_list = ["08"]  # "03", "04", "07", "08"
    heads = [4]

    for no in data_list:
        for i in heads:
            main(no, i)

