# 1. load data
# 根据原论文内容，要：
# transform the .mrc file into np array
# chunked into pairs of overlapping boxes of size 60*60*60 with strides of 30 voxels
# augment random 90 degree rotation
# randomly cropping 48*48*48 box from 60*60*60box

import mrcfile
import utils
import os
import numpy as np


def readMrc2Np(folder_path, map_name):
    mrc_arrays = {}
    for filename in os.listdir(folder_path):
        for map in os.listdir(filename):
            if(os.path.basename(map) == map_name):
                map_path = os.path.join(filename, map)
                try:
                    with mrcfile.open(map_path, mode='r', permissive=True) as mrc:
                        data = mrc.data.astype(np.float32)
                        mrc_arrays[filename] = data
                        print(f"成功读取{filename}, 维度:{data.shape}")
                except Exception as e:
                    print(f"文件 {filename} 读取失败: {str(e)}")
    return mrc_arrays