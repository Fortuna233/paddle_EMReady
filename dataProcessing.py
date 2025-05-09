import os
from utils import *
from sklearn.model_selection import train_test_split



depoFolder = "/home/ty/training_and_validation_sets/depoFiles"
# depoFolder = "/data1/ryi/training_and_validation_sets/depoFiles"
simuFolder = "/home/ty/training_and_validation_sets/simuFiles"
# simuFolder = "/data1/ryi/training_and_validation_sets/simuFiles"
save_dir="../training_and_validation_sets/datasets"
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



# n_chunks, i = 0, 0
# for depoFile, simuFile in zip(depoList, simuList):
#     if(os.path.getsize(depoFile) > 1024 * 1024 * 128 or os.path.getsize(simuFile) > 1024 * 1024 * 128):
#         continue
#     n_chunks += split_and_save_tensor(depoFile, simuFile, save_dir)
#     i += 1 
#     print(f'processing: {i}/{n_maps}')


# for f in os.listdir(save_dir):
#     if f.endswith('.npy'):
#         file_path = os.path.join(save_dir, f)
#         chunk = np.load(file_path)
#         if not (np.sum(chunk[0]) > 0 and np.sum(chunk[1])):
#             print(np.sum(chunk[0]), np.sum(chunk[1]))
#             os.remove(file_path)
#             print(f"remove {file_path}")




chunks_file = [os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.endswith('.npz')]
trainData, valiData = train_test_split(chunks_file, test_size=0.25, random_state=42)
print(trainData[0])
print(valiData[0])
train_iter = data_iter(trainData, batch_size=32, shuffle=True)
vali_iter = data_iter(valiData, batch_size=32, shuffle=False)

    
print(f"训练集样本数: {len(trainData)}")
print(f"测试集样本数: {len(valiData)}")