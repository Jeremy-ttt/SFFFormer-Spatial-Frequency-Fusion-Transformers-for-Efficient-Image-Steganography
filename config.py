from dataclasses import dataclass
import torch

# 以类的方式定义参数
@dataclass
class Args:
    # training config

    # model config 
    image_size_train = 256
    image_size_test_single = 256
    num_secret = 1

    # optimer config
    lr = 2e-4
    warm_up_epoch = 20
    warm_up_lr_init = 1e-6

    # dataset
    DIV2K_path = r'D:\1PaperExperimentCode\datasets\DIV2K_HR'     # /home/whq135/dataset/DIV2K_train_HR
    COCO_path = '/root/autodl-fs/datasets/sub_coco'

    single_batch_size = 12
    
    epochs = 6000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    val_freq = 10
    save_freq = 50
    train_next = 0
    use_model = 'SFFFormer'
    input_dim = 3
    
    norm_train = 'clamp'
    output_act = None
    path='/root/autodl-fs/exp1/SFFFormer'
    model_name='SFFFormer'