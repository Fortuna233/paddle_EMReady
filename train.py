
import os
import random
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
from pytorch_msssim import ssim
from scunet import SCUNet
from utils import pad_map, chunk_generator, parse_map, get_batch_from_generator

from torch import FloatTensor as FT
from torch.autograd import Variable as V


depoFolder = "/home/tyche/training_and_validation_sets/depoFiles"
simuFolder = "/home/tyche/training_and_validation_sets/simuFiles"
batch_size = 8
apix = 4
num_epochs = 300

def get_all_files(directory):
    file_list = list()
    for file in os.listdir(directory):
        file_list.append(f"{directory}/{file}")
    return file_list


depoList = get_all_files(depoFolder)
simuList = get_all_files(simuFolder)
depoList.sort()
simuList.sort()



def mrc2map(mrcfile, apix):
    map, origin, nxyz, voxel_size, nxyz_origin = parse_map(mrcfile, ignorestart=False, apix=apix)
    print(f"origin: {origin}, nxyz: {nxyz}, nxyz_origin: {nxyz_origin}")
    print(f"# Original map dimensions: {nxyz_origin}")
    try:
        assert np.all(np.abs(np.round(origin / voxel_size) - origin / voxel_size) < 1e-4)

    except AssertionError:
        origin_shift =  ( np.round(origin / voxel_size) - origin / voxel_size ) * voxel_size
        map, origin, nxyz, voxel_size, _ = parse_map(mrcfile, ignorestart=False, apix=apix, origin_shift=origin_shift)
        assert np.all(np.abs(np.round(origin / voxel_size) - origin / voxel_size) < 1e-4)
        print(f"origin: {origin}, nxyz: {nxyz}, nxyz_origin: {nxyz_origin}")   

    nxyzstart = np.round(origin / voxel_size).astype(np.int64)
    print(f"# Map dimensions at {apix} Angstrom grid size: {nxyz}")
    maximum = np.percentile(map[map > 0], 99.999)
    return map, nxyzstart, maximum




# def alignment(depoPadded, simuPadded):

    

# 同时对depochunks、simuchunks进行图像增广
def transform(tensor1, tensor2, outsize=48):
    N = tensor1.shape[0]
    axes_options=[(0,1), (1, 2), (0, 2)]
    nx, ny, nz = tensor1.shape[1:4]
    newx, newy, newz = outsize, outsize, outsize
    output1 = torch.zeros(N, 48, 48, 48)
    output2 = torch.zeros(N, 48, 48, 48)
    for i in range(N):
        k = random.choice([1, 2, 3]) 
        rotated1 = torch.rot90(tensor1[i], k=k, dims=random.choice(axes_options))
        rotated2 = torch.rot90(tensor2[i], k=k, dims=random.choice(axes_options))
        startX = random.randint(0, nx-newx)
        startY = random.randint(0, ny-newy)
        startZ = random.randint(0, nz-newz)
        cropped1 = rotated1[startX:startX+outsize, startY:startY+outsize, startZ:startZ+outsize]
        cropped2 = rotated2[startX:startX+outsize, startY:startY+outsize, startZ:startZ+outsize]
        output1[i] = cropped1
        output2[i] = cropped2
    del tensor1
    del tensor2
    torch.cuda.empty_cache()
    return output1, output2


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



print(depoList[1])
depoPadded, depoStart, depoMax = mrc2map(depoList[-1], 1.0)
depoPadded = depoPadded[::5, ::5, ::5]
print(f"shape: {depoPadded.shape}")
x, y, z = np.where(depoPadded > 0)  # 筛选高值区域
values = depoPadded[x, y, z]

fig = go.Figure(data=go.Scatter3d(
    x=x, y=y, z=z, mode='markers',
    marker=dict(size=5, color=values, colorscale='Viridis', opacity=0.8)
))
fig.update_layout(scene=dict(aspectmode='cube'))
fig.show()


simuPadded, simuStart, simuMax = mrc2map(simuList[-1], 1.0)
simuPadded = simuPadded[::5, ::5, ::5]
print(f"shape: {simuPadded.shape}")
x, y, z = np.where(simuPadded > 0)  # 筛选高值区域
values = simuPadded[x, y, z]

fig = go.Figure(data=go.Scatter3d(
    x=x, y=y, z=z, mode='markers',
    marker=dict(size=5, color=values, colorscale='Viridis', opacity=0.8)
))
fig.update_layout(scene=dict(aspectmode='cube'))
fig.show()


'''
model.train()
for depoFile, simuFile in zip(depoList, simuList):
    if(os.path.getsize(depoFile) > 1024 * 1024 * 512 or os.path.getsize(simuFile) > 1024 * 1024 * 512):
        continue
    print(depoFile)
    depoMap, depoStart, depoMax = mrc2map(depoFile, 1.0)
    print(f"shape: {depoMap.shape}")
    print(simuFile)
    simuMap, simuStart, simuMax = mrc2map(simuFile, 1.0)
    print(f"shape: {simuMap.shape}")

  
    depoPadded = pad_map(depoMap, 60)
    del depoMap
    simuPadded = pad_map(simuMap, 60)
    del simuMap
    #两个map不一样大，如何align，同时生成一样大的且对应的map
    for epoch in range(num_epochs):
        train_loss = 0
        depo_generator = chunk_generator(depoPadded, depoMax, 60, 30)
        simu_generator = chunk_generator(simuPadded, simuMax, 60, 30)
        while True:
            positions_depo, depo_chunks = get_batch_from_generator(depo_generator, batch_size, dtype=np.float32)
            positions_simu, simu_chunks = get_batch_from_generator(simu_generator, batch_size, dtype=np.float32)
            if depo_chunks.shape != simu_chunks.shape:
                continue
            
            if depo_chunks.shape[0] == 0:
                break
            depo_chunks = torch.from_numpy(depo_chunks).float()
            simu_chunks = torch.from_numpy(simu_chunks).float()
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

        print(f"epoch:{epoch} depofile:{depoFile} train_loss:{train_loss}")
'''



