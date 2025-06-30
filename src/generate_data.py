import os
import numpy as np
from utils.utils_dataprocessing import get_all_files, align_and_split_tensor
from constants import depoMapFolder, simuMapFolder, datasetsFolder, box_size, stride


depo_map_list = get_all_files(depoMapFolder)
simu_map_list = get_all_files(simuMapFolder)
print(f"lenth of depo_map_list:{len(depo_map_list)}")
print(f"lenth of simu_map_list:{len(simu_map_list)}")


# for i, (depoFile, simuFile) in enumerate(zip(depo_map_list, simu_map_list)):
#     result = align_and_split_tensor(depoFile=depoFile, simuFile=simuFile,  datasetsFolder=datasetsFolder, map_index=i, box_size=box_size, stride=stride, mode='3d')
#     print(f'processing: {i + 1}/{len(depo_map_list)}')


# with os.scandir(datasetsFolder) as entries:
#     count = sum(1 for _ in entries)
# print(f"total number of chunks: {count}")
def check_and_delete_zero_npz(folder_path):

    if not os.path.exists(folder_path):
        print(f"error'{folder_path}' not exist")
        return
    
    # 获取文件夹中所有npz文件
    npz_files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
    
   
    total_files = len(npz_files)
    deleted_files = 0

    for npz_file in npz_files:
        file_path = os.path.join(folder_path, npz_file)
        
        try:
            with np.load(file_path) as data:
                all_zero = True

                for key in data:
                    array = data[key]
                    if not np.all(array == 0):
                        all_zero = False
                        break
                
                if all_zero:
                    os.remove(file_path)
                    deleted_files += 1
                    print(f"delete all 0 array: {npz_file}")
        
        except Exception as e:
            print(f"error when processing {npz_file}: {e}")
    


# 使用示例
if __name__ == "__main__":
    check_and_delete_zero_npz(datasetsFolder)