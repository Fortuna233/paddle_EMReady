import os
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from constants import datasetsFolder


def check_and_delete_zero_npz(folder_path):
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在")
        return
    
    # 获取文件夹中所有npz文件
    npz_files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
    
    total_files = len(npz_files)
    if total_files == 0:
        print("未找到NPZ文件")
        return
    
    print(f"找到 {total_files} 个NPZ文件，开始并行处理...")
    
    # 创建部分应用函数，固定folder_path参数
    worker_func = partial(process_npz_file, folder_path=folder_path)
    
    # 使用CPU核心数减1的进程池（保留一个核心给系统）
    with Pool(processes=max(1, cpu_count() - 1)) as pool:
        results = pool.map(worker_func, npz_files)
    
    # 统计删除的文件数
    deleted_files = sum(results)
    print(f"处理完成：总共 {total_files} 个文件，删除了 {deleted_files} 个全零数组文件")

def process_npz_file(npz_file, folder_path):
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
                print(f"删除全零数组文件：{npz_file}")
                return 1  # 表示已删除
    except Exception as e:
        print(f"处理文件 {npz_file} 时出错：{e}")
    
    return 0  # 表示未删除

if __name__ == "__main__":

    check_and_delete_zero_npz(datasetsFolder)    