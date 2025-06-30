import os
from torch.cuda import set_device
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size, gpu_ids=None):
    if gpu_ids is not None:
        set_device(gpu_ids[rank])
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def prepare_dataloader(dataset, batch_size, is_train=True):
    if is_train:
        sampler = DistributedSampler(dataset, shuffle=True)
    else:
        sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, sampler=sampler)
    return dataloader


def create_ddp_model(rank, model):
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    return ddp_model
