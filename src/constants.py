from models.scunet import SCUNet


mode = '3d'
world_size = 3
num_epochs = 10
batch_size = 8
lr = 5e-4
vali_ratio = 0.1
box_size = 60
stride = 30
stride_inference = 12
accumulations_steps = 1

depoMapFolder = '/data1/ryi/training_and_validation_sets/depoMaps'
simuMapFolder = '/data1/ryi/training_and_validation_sets/simuMaps'
datasetsFolder = '/data1/ryi/paddle_EMReady-main/data/datasets'
paramsFolder = '/data1/ryi/paddle_EMReady-main/data/params'
predictionsFolder = '/data1/ryi/paddle_EMReady-main/data/predictions'
logsFolder = '/data1/ryi/paddle_EMReady-main/data/logs'
resultFolder = '/data1/ryi/paddle_EMReady-main/data/results'


def create_model():
    return SCUNet(
        in_nc=1,
        config=[2,2,2,2,2,2,2],
        dim=32,
        drop_path_rate=0.0,
        input_resolution=48,
        head_dim=16,
        window_size=3,
    )