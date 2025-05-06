
import os
from math import ceil
import numpy as np
from torchvision import models
import torch
import torch.nn as nn
from pytorch_msssim import ssim
from scunet import SCUNet
from utils import *
from torch import FloatTensor as FT
from torch.autograd import Variable as V


# depoFolder = "/home/ty/training_and_validation_sets/depoFiles"
depoFolder = "/data1/ryi/training_and_validation_sets/depoFiles"
# simuFolder = "/home/ty/training_and_validation_sets/simuFiles"
simuFolder = "/data1/ryi/training_and_validation_sets/simuFiles"
save_dir="datasets"
batch_size = 64
apix = 1
num_epochs = 300


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

n_chunks = 0
for depoFile, simuFile in zip(depoList, simuList):
    if(os.path.getsize(depoFile) > 1024 * 1024 * 512 or os.path.getsize(simuFile) > 1024 * 1024 * 512):
        continue
    n_chunks += split_and_save_tensor(depoFile, simuFile, save_dir) 
devices = try_all_gpus()

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
model = model.to(devices[0])
model = nn.DataParallel(model)

# 定义损失函数
def loss(pred, target):
    smooth_l1_loss = nn.SmoothL1Loss(beta=1.0, reduction='mean')
    return smooth_l1_loss(pred, target) + 1 - ssim(pred, target, data_range=1.0, size_average=True)
    

# 定义trainer
trainer = torch.optim.Adam(
    model.parameters(),
    lr=0.0005,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,
    amsgrad=False
)


model.train()

for epoch in range(num_epochs):
    train_loss = 0
    cur_steps = 0
    for batch in data_iter(save_dir=save_dir, batch_size=batch_size): 
        depo_chunks, simu_chunks = batch[0], batch[1]
        if depo_chunks.shape[0] == 0 or simu_chunks.shape[0] == 0:
            continue
        depo_chunks = torch.from_numpy(depo_chunks)
        simu_chunks = torch.from_numpy(simu_chunks)
        #保证depo和simu这俩map对每个chunk的操作完全一致，即密度能完全对应上
        depo_chunks, simu_chunks = transform(depo_chunks, simu_chunks)
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
        print(f"processing: {cur_steps} / {n_chunks}, loss: {l:.5f}")
    print(f"epoch:{epoch} depofile:{depoFile} train_loss:{train_loss:.5f}")