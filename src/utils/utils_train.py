import os
import time
import random
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split


from pytorch_msssim import ssim
import torch
import torch.utils
from torch import nn
import torchvision.transforms as T
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR

from utils.utils_ddp import *
from utils.utils_dataprocessing import get_all_files



def try_all_gpus():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


class myDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list


    def __len__(self):
        return len(self.file_list)
    

    def __getitem__(self, index):
        file_path = self.file_list[index]
        data = np.load(file_path)['combined']
        return torch.from_numpy(data)
        

def get_dataset(save_path, batch_size):
    chunks_file = [os.path.join(save_path, f) for f in os.listdir(save_path) if f.endswith('.npz')]
    trainData, valiData = train_test_split(chunks_file, test_size=0.1, random_state=42)
    trainSet = myDataset(trainData, is_train=True)
    valiSet = myDataset(valiData, is_train=False)
    train_iter = DataLoader(trainSet, batch_size=batch_size, shuffle=True, num_workers=4 * torch.cuda.device_count(), pin_memory=True, prefetch_factor=2)
    vali_iter = DataLoader(valiSet, batch_size=batch_size, shuffle=False, num_workers=4 * torch.cuda.device_count(), pin_memory=True, prefetch_factor=2)
    return train_iter, vali_iter


def loss(pred, target):
    # 检查输入是否包含NaN
    if torch.isnan(pred).any() or torch.isnan(target).any():
        print("NaN in input!")
        return torch.tensor(0.0, device=pred.device)
    
    smooth_l1 = nn.SmoothL1Loss()
    ssim_val = ssim(pred, target, data_range=1.0, size_average=True)
    
    # 检查SSIM计算结果
    if torch.isnan(ssim_val):
        print("SSIM NaN!")
        return smooth_l1(pred, target)
    
    return smooth_l1(pred, target) + 1 - ssim_val


def transform(tensor1, tensor2, outsize=48):
    # batch_mask = (torch.sum(tensor1 > 0, dim=(1, 2, 3)) > 0) | \
    #              (torch.sum(tensor2 > 0, dim=(1, 2, 3)) > 0)
    # tensor1 = tensor1[batch_mask]
    # tensor2 = tensor2[batch_mask]
    
    # if tensor1.numel() == 0:
    #     return torch.zeros(0, outsize, outsize, outsize, device=tensor1.device), \
    #            torch.zeros(0, outsize, outsize, outsize, device=tensor2.device)
    
    # 获取批次大小和原始尺寸
    N, nx, ny, nz = tensor1.shape
    
    # 计算安全的随机起始点（避免裁剪超出边界）
    max_start = np.array([nx, ny, nz]) - outsize
    starts = torch.randint(0, max_start.min(), (N, 3), device=tensor1.device)
    
    # 为每个样本选择随机旋转轴和旋转次数
    axes_options = np.array([(0, 1), (1, 2), (0, 2)])
    axes_indices = torch.randint(0, 3, (N,), device=tensor1.device)
    k_rotations = torch.randint(0, 4, (N,), device=tensor1.device)
    
    # 创建输出张量
    cropped1 = torch.zeros(N, outsize, outsize, outsize, device=tensor1.device, dtype=tensor1.dtype)
    cropped2 = torch.zeros(N, outsize, outsize, outsize, device=tensor2.device, dtype=tensor2.dtype)
    
    # 向量化裁剪和旋转操作
    for i in range(N):
        # 获取当前样本的旋转轴
        axes = tuple(axes_options[axes_indices[i].item()])  # 关键修改点
        
        # 应用旋转
        tensor1[i] = torch.rot90(tensor1[i], k_rotations[i], dims=axes)
        tensor2[i] = torch.rot90(tensor2[i], k_rotations[i], dims=axes)
        
        # 裁剪
        s = starts[i]
        cropped1[i] = tensor1[i, s[0]:s[0]+outsize, s[1]:s[1]+outsize, s[2]:s[2]+outsize]
        cropped2[i] = tensor2[i, s[0]:s[0]+outsize, s[1]:s[1]+outsize, s[2]:s[2]+outsize]
    
    return cropped1, cropped2


def train(rank, world_size, model, paramsFolder, datasetsFolder, logsFolder, num_epochs, batch_size, lr, accumulation_steps):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"num_parameters: {total_params}")
    chunks_file = [os.path.join(datasetsFolder, f) for f in os.listdir(datasetsFolder) if f.endswith('.npz')]
    trainSet = myDataset(chunks_file)
    train_iter = prepare_dataloader(trainSet, batch_size=batch_size, is_train=True)
    
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv3d or type(m) == nn.Conv2d:  
            nn.init.xavier_uniform_(m.weight)
    current_epochs = len(get_all_files(paramsFolder))
    model.apply(init_weights)
    trainer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False)
    scheduler = OneCycleLR(trainer, max_lr=lr, total_steps=int(len(train_iter)*(num_epochs - current_epochs)/accumulation_steps), pct_start=0.3, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95,div_factor=25,final_div_factor=1e5)
    scaler = GradScaler()

    latest_checkpoint = get_latest_checkpoint(paramsFolder)
    if latest_checkpoint:
        current_epoch, model, trainer, scheduler = load_checkpoint(
            model, trainer, scheduler, latest_checkpoint, device=device
        )
        current_epoch += 1  # 从下一个epoch开始
    else:
        current_epoch = 0
        print("No checkpoint found, starting from scratch")

    model = create_ddp_model(rank=rank, model=model)  

    train_Loss = []
    starttime = time.time()
    local_now = datetime.now()
    if rank==0:
        print(local_now)
    for epoch in range(current_epoch, num_epochs):
        train_iter.sampler.set_epoch(epoch)
        train_loss = 0
        loss_batch = 0
        model.train()
        for i, combined_chunk in enumerate(train_iter):
            X, Y = transform(combined_chunk[:, 0], combined_chunk[:, 1])
            if X.shape[0] == 0 or Y.shape[0] == 0:
                continue
            batch_size, *spatial_dims = X.shape
            X = X.reshape(batch_size, 1, *spatial_dims).to(rank)
            Y = Y.reshape(batch_size, 1, *spatial_dims).to(rank)
            with autocast(device_type='cuda'):
                pred = model(X).to(rank)
                l = loss(pred, Y)
                train_loss += l.item()
                loss_batch += l.item() / accumulation_steps
                del X, Y, pred
            scaler.scale(l).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if (i + 1) % accumulation_steps == 0:
                scaler.step(trainer)
                scaler.update()
                scheduler.step()
                trainer.zero_grad()
                Time = time.time() - starttime
                log_msg = f"[epoch: {epoch}] [processing: {i + 1}/{len(train_iter)}] [loss: {loss_batch}] [lr: {trainer.param_groups[0]['lr']}] [Time: {Time:.2f}s]"
                if rank == 0:
                    print(log_msg)
                    loss_batch = 0
                    with open(f"{logsFolder}/output_{local_now}.log", "a") as file:
                        file.write(log_msg + "\n")
            
        epoch_loss = train_loss / len(train_iter)
        train_Loss.append(epoch_loss)

        
        log_msg = f"[epoch: {epoch}] [processing: {i + 1}/{len(train_iter)}] [train_loss: {train_Loss[-1]}] [lr: {trainer.param_groups[0]['lr']}] [Time: {Time:.2f}s]"
        if rank == 0:
            print(log_msg)
            print(f"checkPoint_{epoch}")
            print("=================================================================================================================")
            with open(f"{logsFolder}/output_{local_now}.log", "a") as file:
                file.write(log_msg + "\n")  
            save_checkpoint(epoch=epoch, model=model, optimizer=trainer, scheduler=scheduler, save_dir=paramsFolder, filename=f"checkPoint_{epoch}.pth")
    cleanup()


def save_checkpoint(epoch, model, optimizer, scheduler, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
    }

    save_path = os.path.join(save_dir, filename)
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")
    return save_path


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device='cpu'):
    """从检查点加载模型、优化器和调度器状态"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # 将优化器参数移到指定设备
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    current_epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {checkpoint_path}")
    return current_epoch, model, optimizer, scheduler


def get_latest_checkpoint(save_dir):
    """获取目录中最新的检查点文件路径"""
    if not os.path.exists(save_dir):
        return None
    checkpoint_files = [f for f in os.listdir(save_dir) if f.endswith('.pth')]
    if not checkpoint_files:
        return None
    checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(save_dir, x)), reverse=True)
    return os.path.join(save_dir, checkpoint_files[0])    
