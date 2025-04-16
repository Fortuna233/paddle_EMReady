
import os
from math import ceil
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pytorch_msssim import ssim
from scunet import SCUNet
from utils import pad_map, chunk_generator, mrc2map, align, transform, get_batch_from_generator
from torch import FloatTensor as FT
from torch.autograd import Variable as V


depoFolder = "/home/tyche/training_and_validation_sets/depoFiles"
simuFolder = "/home/tyche/training_and_validation_sets/simuFiles"
batch_size = 8
apix = 1
num_epochs = 30


# 数据预处理
def get_all_files(directory):
    file_list = list()
    for file in os.listdir(directory):
        file_list.append(f"{directory}/{file}")
    return file_list


depoList = get_all_files(depoFolder)
simuList = get_all_files(simuFolder)
depoList.sort()
simuList.sort()


# 输入为torch张量batch_size*60*60*60
model = SCUNet(
    in_nc=1,
    config=[2,2,2,2,2,2,2],
    dim=32,
    drop_path_rate=0.0,
    input_resolution=48,
    head_dim=16,
    window_size=3,
)
torch.cuda.empty_cache()
model = model.cuda()


# 定义损失函数
def loss(pred, target):
    smooth_l1_loss = nn.SmoothL1Loss(beta=1.0, reduction='mean')
    return smooth_l1_loss(pred, target) + 1 - ssim(pred, target, data_range=1.0,    size_average=True)
    

# 定义trainer
trainer = torch.optim.Adam(
    model.parameters(),
    lr=0.0005,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,
    amsgrad=False
)



# print(depoList[1])
# depoMap = mrc2map(depoList[1], 1.0)
# # 降采样
# depoMap = depoMap[::20, ::20, ::20]
# print(f"shape: {depoMap.shape}")
# x, y, z = np.where(depoMap)  # 筛选高值区域
# print(x.shape)
# values = depoMap[x, y, z]
# print(f"shape of values: {values.shape}")

# # fig = go.Figure(data=go.Scatter3d(
# #     x=x, y=y, z=z, mode='markers',
# #     marker=dict(size=5, color=values, colorscale='Viridis', opacity=0.8)
# # ))
# # fig.update_layout(scene=dict(aspectmode='cube'))
# # fig.show()


# # plt.hist(values, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.7)
# # plt.xlabel('interval')
# # plt.ylabel('likelihood')
# # plt.title('frequency histogram')
# # plt.show()


# print(simuList[1])
# simuPadded = mrc2map(simuList[1], 1.0)
# # 降采样
# simuPadded = simuPadded[::20, ::20, ::20]
# print(f"shape: {simuPadded.shape}")
# x, y, z = np.where(simuPadded)  # 筛选高值区域
# values = simuPadded[x, y, z]
# print(f"shape of values: {values.shape}")
# # fig = go.Figure(data=go.Scatter3d(
# #     x=x, y=y, z=z, mode='markers',
# #     marker=dict(size=5, color=values, colorscale='Viridis', opacity=0.8)
# # ))
# # fig.update_layout(scene=dict(aspectmode='cube'))
# # fig.show()


# # plt.hist(values, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.7)
# # plt.xlabel('interval')
# # plt.ylabel('likelihood')
# # plt.title('frequency histogram')
# # plt.show()


model.train()
for depoFile, simuFile in zip(depoList, simuList):
    if(os.path.getsize(depoFile) > 1024 * 1024 * 512 or os.path.getsize(simuFile) > 1024 * 1024 * 512):
        continue
    print(depoFile)
    depoMap = mrc2map(depoFile, 1.0)
    print(simuFile)
    simuMap = mrc2map(simuFile, 1.0)
    
    #对齐
    depoMap = align(depoMap, simuMap)
    depoPadded = pad_map(depoMap, 60)
    del depoMap
    simuPadded = pad_map(simuMap, 60)
    del simuMap

    for epoch in range(num_epochs):
        train_loss = 0
        depo_generator = chunk_generator(depoPadded, 60, 30)
        simu_generator = chunk_generator(simuPadded, 60, 30)
        ncx, ncy, ncz = [ceil(depoPadded.shape[i] / 30) for i in range(3)]
        total_steps = float(ncx * ncy * ncz)
        cur_steps = 0
        while True:
            depo_positions, depo_chunks = get_batch_from_generator(depo_generator, batch_size, dtype=np.float32)
            simu_positions, simu_chunks = get_batch_from_generator(simu_generator, batch_size, dtype=np.float32)
            depo_chunks = torch.from_numpy(depo_chunks)
            simu_chunks = torch.from_numpy(simu_chunks)     

            if depo_chunks.shape[0] == 0 or simu_chunks.shape[0] == 0:
                break
            # 保证depo和simu这俩map对每个chunk的操作完全一致，即密度能完全对应上
            # 去除全零chunk（若有一全为零则同时去除两个的）
            print(f"depo_chunks.shape: {depo_chunks.shape}")
            print(f"simu_chunks.shape: {simu_chunks.shape}")
            depo_chunks, simu_chunks = transform(depo_chunks, simu_chunks)
            print(f"depo_chunks.shape: {depo_chunks.shape}")
            print(f"simu_chunks.shape: {simu_chunks.shape}")
            X = V(FT(depo_chunks), requires_grad=True).view(-1, 1, 48, 48, 48)
            X = X.cuda()
            simu_chunks = simu_chunks.cuda()
            y_pred = model(X).view(-1, 48, 48, 48)           
            l = loss(y_pred, simu_chunks)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            train_loss += l
            cur_steps += len(depo_chunks)
            print(f"train_loss: {l:.2f}")
            print(f"processing: {cur_steps} / {total_steps}")

        print(f"epoch:{epoch} depofile:{depoFile[-18][:4]} train_loss:{train_loss:.2f}")
        # 保存检查点
    checkpoint = {'mrcfile': depoFile[-18:][:4],
                  'model_state_dict': model.state_dict(),'optimizer_state_dict': trainer.state_dic(),
                  'loss': train_loss,}
    torch.save(checkpoint, 'checkpoint.pth')


torch.save(model.state_dict(), 'model_weights.pth')

