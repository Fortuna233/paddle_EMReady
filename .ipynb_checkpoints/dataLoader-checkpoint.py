
# load data
# 根据原论文内容，要：
# transform the .mrc file into np array
# chunked into pairs of overlapping boxes of size 60*60*60 with strides of 30 voxels
# augmentation:
# random 90 degree rotation
# randomly cropping 48*48*48 box from 60*60*60box

import os
import random
import mrcfile
import numpy as np
import interp_back
import torch
import torchvision
from torch.utils import data
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
from utils import pad_map, chunk_generator, parse_map

depoFolder = "/home/tyche/training_and_validation_sets/depoFiles"
simuFolder = "/home/tyche/training_and_validation_sets/simuFiles"
box_size = 60
stride = 30



def indices_of_map(paddedmap, box_size, stride, dtype=np.float32, padding=0.0):
    assert stride <= box_size
    map_shape = np.shape(paddedmap)
    map_shape -= 2 * box_size
    indices = list()
    start_point = box_size - stride
    cur_x, cur_y, cur_z = start_point, start_point, start_point
    while (cur_z + stride < map_shape[2] + box_size):
        # next_chunk = padded_map[cur_x:cur_x + box_size, cur_y:cur_y + box_size, cur_z:cur_z + box_size]
        indices.append([cur_x, cur_y, cur_z])
        cur_x += stride
        if (cur_x + stride >= map_shape[0] + box_size):
            cur_y += stride
            cur_x = start_point # Reset X
            if (cur_y + stride  >= map_shape[1] + box_size):
                cur_z += stride
                cur_y = start_point # Reset Y
                cur_x = start_point # Reset X
    # n_chunks = len(indices)
    return indices


def mrc2paddmap(mrcFile):
    map, _, _, _, _ = parse_map(mrcFile, ignorestart=False)
    maximum = np.percentile(map[map > 0], 99.999)
    map = np.where(map > 0, map / maximum, 0)
    padded_map = pad_map(map, 60, dtype=np.float32, padding=0.0)
    del map
    return padded_map

        
def get_chunks(padded_map, box_size, batch_indices):
    x, y, z = batch_indices[:, 0], batch_indices[:, 1], batch_indices[:, 2]
    idx = x[:, None, None, None] + torch.arange(box_size)[None, :, None, None]
    idy = y[:, None, None, None] + torch.arange(box_size)[None, None, :, None]
    idz = z[:, None, None, None] + torch.arange(box_size)[None, None, None, :]
    return padded_map[idx, idy, idz]



    

def transform(tensor, outsize=48):
    N = tensor.shape[0]
    axes_options=[(0,1), (1, 2), (0, 2)]
    nx, ny, nz = tensor.shape[1:4]
    newx, newy, newz = outsize, outsize, outsize
    output = torch.zeros(N, 48, 48, 48, device=tensor.device)
    for i in range(N):
        k = random.choice([1, 2, 3]) 
        rotated = torch.rot90(tensor[i], k=k, dims=random.choice(axes_options))
        startX = random.randint(0, nx-newx)
        startY = random.randint(0, ny-newy)
        startZ = random.randint(0, nz-newz)
        cropped = rotated[startX:startX+outsize, startY:startY+outsize, startZ:startZ+outsize]
        output[i] = cropped
    del tensor
    torch.cuda.empty_cache()
    return output


def get_all_files(directory):
    file_list = list()
    for file in os.listdir(directory):
        file_list.append(f"{directory}/{file}")
    return file_list


depoList = get_all_files(depoFolder)
simuList = get_all_files(simuFolder)
depoList.sort()
simuList.sort()


def data_iter(batch_size, depoList, simuList):
    for depofile, simufile in zip(depoList, simuList):
        depo_padded = mrc2paddmap(depofile)
        simu_padded = mrc2paddmap(simufile)
        depo_padded =  F.interpolate(depo_padded, size=(simu_padded.shape), mode='trilinear', align_corners=False)
        depoIndices = indices_of_map(depo_padded, box_size=box_size, stride=stride)
        simuIndices = indices_of_map(simu_padded, box_size=box_size, stride=stride)
        assert len(depoIndices) == len(simuIndices)
        num_examples = len(depoIndices)
        indices = list(range(num_examples))
        random.shuffle(list(range(len(depoIndices))))
        print(num_examples)
        print(depo_padded.shape)
        print(simu_padded.shape)
        for i in range(0, num_examples, batch_size):
            batch_indices = indices[i : min(i + batch_size, num_examples)]
            print(batch_indices)
            print(depoIndices[batch_indices], simuIndices[batch_indices])
            depo_chunks = get_chunks(depo_padded, 48, depoIndices[batch_indices])
            simu_chunks = get_chunks(simu_padded, 48, simuIndices[batch_indices])
            yield depo_chunks, simu_chunks
            


batch_size = 2
for depo_chunks, simu_chunks in data_iter(batch_size, depoList, simuList):
    print(depo_chunks, '\n', simu_chunks)
    break


