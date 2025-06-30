
import torch.multiprocessing as mp
from constants import world_size, paramsFolder, datasetsFolder, logsFolder, num_epochs, batch_size, lr, accumulations_steps, create_model
from utils.utils_train import train

if __name__ == "__main__":
    model = create_model()
    mp.spawn(train, args=(world_size, model, paramsFolder, datasetsFolder, logsFolder, num_epochs, batch_size, lr, accumulations_steps), nprocs=world_size, join=True)