import os
import mrcfile
import torch
import numpy as np
import multiprocessing
from functools import partial
from itertools import product
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.utils import parse_map, mrc2map

def get_all_files(directory):
    file_list = []
    for file in os.listdir(directory):
        file_list.append(f"{directory}/{file}")
    return sorted(file_list)


def align_and_split_tensor(depoFile, simuFile, datasetsFolder, map_index, box_size=48, stride=12, mode='3d'):
    if not os.path.exists(datasetsFolder):
        os.makedirs(datasetsFolder)
    
    depo_data, depoMax = mrc2map(depoFile, 1.0)
    simu_data, simuMax = mrc2map(simuFile, 1.0)

    print("start aligning")
    print(f"original map shape: depo={depo_data.shape}, simu={simu_data.shape}")
    depo_data, simu_data = align(depo_data, simu_data)
    print(f"aligned depo_map shape: {depo_data.shape} (aligned simu_map shape: {simu_data.shape})")
    print("normalization")
    depo_data = depo_data.clip(min=0.0, max=depoMax) / depoMax
    simu_data = simu_data.clip(min=0.0, max=simuMax) / simuMax
    depo_padded = pad_volume(depo_data, box_size)
    simu_padded = pad_volume(simu_data, box_size)
    map_shape = depo_data.shape
    del depo_data, simu_data
    chunk_coords_generator = generate_chunk_coords(map_shape, box_size, stride, mode)
    
    num_workers = multiprocessing.cpu_count()
    process_args = {
        'depo_padded': depo_padded,
        'simu_padded': simu_padded,
        'datasetsFolder': datasetsFolder,
        'map_index': map_index,
    }
    
    process_func = partial(process_aligned_chunks, **process_args)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_func, coords) for coords in chunk_coords_generator]
        for future in as_completed(futures):
            try:
                chunk_filepath, chunk_shape = future.result()
                print(f"save chunk: {chunk_filepath}, size={chunk_shape}")
            except Exception as e:
                print(f"error when processing chunks: {e}")
    
    return map_shape


def load_and_normalize(map_file, minPercent, maxPercent, mode):

    with mrcfile.open(map_file, mode='r') as mrc:
        map_data = np.asarray(mrc.data.copy(), dtype=np.float32)
    
    if len(map_data.shape) == 2:
        map_data = map_data.reshape(-1, *map_data.shape)
    map_data = normalize(map_data, minPercent=minPercent, maxPercent=maxPercent, mode=mode)
    return map_data


def align(depoMap, simuMap):
    padded_shape = [max(depoMap.shape[0], simuMap.shape[0]),
                    max(depoMap.shape[1], simuMap.shape[1]),
                    max(depoMap.shape[2], simuMap.shape[2])]
    # 对两个信号进行零填充（将原始数据放在左上角）
    depo_padded = np.zeros(padded_shape, dtype=depoMap.dtype)
    depo_padded[:depoMap.shape[0], :depoMap.shape[1], :depoMap.shape[2]] = depoMap
    simu_padded = np.zeros(padded_shape, dtype=simuMap.dtype)
    simu_padded[:simuMap.shape[0], :simuMap.shape[1], :simuMap.shape[2]] = simuMap
    # 3D FFT
    fft_depo = np.fft.fftn(depo_padded)
    fft_simu = np.fft.fftn(simu_padded)
    # calculate corr
    corr_freq = fft_depo * np.conj(fft_simu)
    # ifftn->real
    corr = np.fft.ifftn(corr_freq).real 

    peak_idx = np.unravel_index(np.argmax(corr), corr.shape)
    dx = peak_idx[0]
    dy = peak_idx[1]
    dz = peak_idx[2]
    print(dx, dy, dz)
    depo_padded = np.roll(depo_padded, shift=-dx, axis=0)
    depo_padded = np.roll(depo_padded, shift=-dy, axis=1)
    depo_padded = np.roll(depo_padded, shift=-dz, axis=2)
    return depo_padded, simu_padded


def pad_to_shape(vol, target_shape):
    padded = np.zeros(target_shape, dtype=vol.dtype)
    offsets = [(target_shape[i] - vol.shape[i]) // 2 for i in range(3)]
    padded[offsets[0]:offsets[0]+vol.shape[0],
           offsets[1]:offsets[1]+vol.shape[1],
           offsets[2]:offsets[2]+vol.shape[2]] = vol
    return padded


def pad_volume(vol, box_size):
    vol_shape = vol.shape
    padded = np.full((vol_shape[0] + 2 * box_size, vol_shape[1] + 2 * box_size, vol_shape[2] + 2 * box_size), 
                    0.0, dtype=np.float32)
    padded[box_size:box_size+vol_shape[0], 
           box_size:box_size+vol_shape[1], 
           box_size:box_size+vol_shape[2]] = vol
    return torch.from_numpy(padded)


def generate_chunk_coords(map_shape, box_size, stride, mode):
    """生成切块坐标"""
    z_range = range(0, map_shape[0] - box_size + 1, stride)
    y_range = range(0, map_shape[1] - box_size + 1, stride)
    x_range = range(0, map_shape[2] - box_size + 1, stride)
    
    for z, y, x in product(z_range, y_range, x_range):
        yield (z, y, x, box_size)


def process_aligned_chunks(chunk_coords, depo_padded, simu_padded, datasetsFolder, map_index):
    z, y, x, size = chunk_coords
    depo_chunk = depo_padded[z:z+size, y:y+size, x:x+size].numpy()
    simu_chunk = simu_padded[z:z+size, y:y+size, x:x+size].numpy()
    combined_chunk = np.stack([depo_chunk, simu_chunk], axis=0)
    chunk_filepath = os.path.join(datasetsFolder, f"{map_index}_z{z}_y{y}_x{x}")
    np.savez_compressed(chunk_filepath, combined=combined_chunk)
    return chunk_filepath, combined_chunk.shape


def normalize(data, minPercent=0, maxPercent=99.999, mode='3d'):
    if mode == '3d' and len(data.shape) == 3:
        non_zero = data[np.nonzero(data)]
        if len(non_zero) > 0:
            vmin = np.percentile(non_zero, minPercent)
            vmax = np.percentile(non_zero, maxPercent)
            data = (data - vmin) / (vmax - vmin + 1e-8)
            data = np.clip(data, 0, 1)
    elif mode == '2d' and len(data.shape) == 3:
        for i in range(data.shape[0]):
            non_zero = data[i, np.nonzero(data[i])]
            if len(non_zero) > 0:
                vmin = np.percentile(non_zero, minPercent)
                vmax = np.percentile(non_zero, maxPercent)
                data[i] = (data[i] - vmin) / (vmax - vmin + 1e-8)
                data[i] = np.clip(data[i], 0, 1)
    return data