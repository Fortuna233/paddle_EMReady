
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
import matplotlib.pyplot as plt

depoFolder = "/home/ty/training_and_validation_sets/depoFiles"
# depoFolder = "/data1/ryi/training_and_validation_sets/depoFiles"
simuFolder = "/home/ty/training_and_validation_sets/simuFiles"
# simuFolder = "/data1/ryi/training_and_validation_sets/simuFiles"
save_dir="datasets"
batch_size = 32
apix = 1
num_epochs = 300


# 数据预处理
depoList, n_depoMaps = get_all_files(depoFolder)
simuList, n_simuMaps = get_all_files(simuFolder)
assert n_depoMaps == n_simuMaps
n_maps = n_depoMaps
depoList.sort()
simuList.sort()



n_chunks, i = 0, 0
for depoFile, simuFile in zip(depoList, simuList):
    if(os.path.getsize(depoFile) <= 1024 * 1024 * 512 and os.path.getsize(simuFile) <= 1024 * 1024 * 512):
        continue
    n_chunks += split_and_save_tensor(depoFile, simuFile, save_dir)
    i += 1 
    print(f'processing: {i}/{n_maps}')


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
model.load_state_dict(torch.load('params/4_params'))
devices = try_all_gpus()
model = model.to(devices[2])
model = torch.nn.DataParallel(model, [2, 3])


# 定义trainer
trainer = torch.optim.Adam(
    model.parameters(),
    lr=0.0005,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,
    amsgrad=False
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trainer, mode='min', factor=0.5, patience=4, min_lr=1e-5)
_, n_chunks = get_all_files(save_dir)

model.train()
train_Loss = []
for epoch in range(num_epochs):
    train_loss = 0
    cur_steps = 0
    for X, y in data_iter(save_dir=save_dir, batch_size=batch_size): 
        # depo_chunks = torch.from_numpy(depo_chunks)
        # simu_chunks = torch.from_numpy(simu_chunks)
        # #保证depo和simu这俩map对每个chunk的操作完全一致，即密度能完全对应上
        # depo_chunks, simu_chunks = transform(depo_chunks, simu_chunks)
        if X.shape[0] == 0 or y.shape[0] == 0:
            continue
        X = V(FT(X), requires_grad=True).view(-1, 1, 48, 48, 48)
        X = X.cuda()
        y = y.cuda()        
        l = loss(model(X).view(-1, 48, 48, 48), y)
        trainer.zero_grad()
        l.backward()
        scheduler.step(l)
        current_lr = trainer.param_groups[0]['lr']
        train_loss += l
        cur_steps += len(X)
        print(f"[processing: {cur_steps} / {n_chunks}] [loss: {l:.5f}] [lr:{current_lr}]")
    train_Loss.append(train_loss)
    print(f"epoch:{epoch} train_loss:{train_loss:.5f}")
    print("===============================================================")
    torch.save(model.module.state_dict(), "checkPoint")

file_path = 'train_Loss.txt'
try:
    with open(file_path, 'w') as file:
        for item in train_Loss:
            file.write(str(item) + '\n')
    print(f"successfully saved {file_path}")
except Exception as e:
    print(f"error in saving: {e}")

plt.plot(range(num_epochs), train_Loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()