import glob
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from natsort import natsorted
from PIL import Image
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
import config
import os


args = config.Args()

# 对数据集图像进行处理
transform = T.Compose([
    T.RandomCrop(128),
    T.RandomHorizontalFlip(),
    T.ToTensor()
])

# 使用 albumentations 库对图像进行处理
transform_A = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.RandomRotate90(),
    A.HorizontalFlip(),
    A.augmentations.transforms.ChannelShuffle(0.3),
    ToTensorV2()
])

transform_A_valid = A.Compose([
    A.CenterCrop(width=256, height=256),
    ToTensorV2()
])

transform_A_test = A.Compose([
    A.SmallestMaxSize(max_size=1024),
    A.CenterCrop(width=1024, height=1024),
    ToTensorV2()
])

transform_A_test_256 = A.Compose([
    A.PadIfNeeded(min_width=256,min_height=256),
    A.CenterCrop(width=256, height=256),
    ToTensorV2()
])

DIV2K_path = args.DIV2K_path
COCO_Path = args.COCO_path
Flickr2K_path = "/home/whq135/dataset/Flickr2K"

batchsize = 12

# dataset


class DIV2K_Dataset(Dataset):
    def __init__(self, transforms_=None, mode='train'):
        self.transform = transforms_
        self.mode = mode
        if mode == 'train':
            self.files = natsorted(
                sorted(glob.glob(DIV2K_path+"/DIV2K_train_HR"+"/*."+"png")))
        else:
            self.files = natsorted(
                sorted(glob.glob(DIV2K_path+"/DIV2K_valid_HR"+"/*."+"png")))

    def __getitem__(self, index):
        img = cv2.imread(self.files[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB
        trans_img = self.transform(image=img)
        item = trans_img['image']
        item = item/255.0
        return item

    def __len__(self):
        return len(self.files)
        
class COCO_Dataset(Dataset):
    def __init__(self, transforms_=None, mode='train'):
        self.transform = transforms_
        self.mode = mode
        if mode == 'train':
            # 后缀改为 jpg，文件夹名改为 train2017
            path = os.path.join(COCO_Path, "train2017/*.jpg")
        elif mode == 'val':
            # 文件夹名对应你划分时的 val2017
            path = os.path.join(COCO_Path, "val2017/*.jpg")
        else:
            # 文件夹名对应你划分时的 test2017
            path = os.path.join(COCO_Path, "test2017/*.jpg")
        self.files = natsorted(glob.glob(path))
        
    def __getitem__(self, index):
        while True:
            idx = index % len(self.files)
            img_path = self.files[idx]
            img = cv2.imread(img_path)
            if img.shape[0] < 256 or img.shape[1] < 256:
                index += 1
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB
            trans_img = self.transform(image=img)
            item= trans_img['image']
            item=item/255.0
            return item

    def __len__(self):
        return len(self.files)
        
# 1. 训练集：直接读取你刚刚切好的小图
class DIV2K_Fast_Dataset(Dataset):
    def __init__(self, transforms_=None, mode='train'):
        if mode == 'train':
            # 指向刚才切好的文件夹
            self.root = '/root/autodl-tmp/datasets/DIV2K_train_256_Patches'
            self.files = glob.glob(os.path.join(self.root, "*.png"))
            self.transform = A.Compose([
                A.RandomRotate90(), # 还可以做旋转翻转
                A.HorizontalFlip(),
                ToTensorV2()
            ])
        else:
            # 2. 验证集：指向原始的大图文件夹
            self.root = '/root/autodl-fs/datasets/DIV2K_HR/DIV2K_valid_HR'
            self.files = glob.glob(os.path.join(self.root, "*.png"))
            # 验证集只做中心裁剪，或者随机裁剪成一张 256
            self.transform = A.Compose([
                A.CenterCrop(256, 256), # 简单快速
                ToTensorV2()
            ])
    def __getitem__(self, index):
        img = cv2.imread(self.files[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB
        trans_img = self.transform(image=img)
        item = trans_img['image']
        item = item/255.0
        return item

    def __len__(self):
        return len(self.files)
        
class Flickr2K_Dataset(Dataset):
    def __init__(self, transforms_=None, mode='train'):
        self.transform = transforms_
        self.mode = mode
        self.files = natsorted(
            sorted(glob.glob(Flickr2K_path+"/Flickr2K_HR"+"/*."+"png")))

    def __getitem__(self, index):
        img = cv2.imread(self.files[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB
        trans_img = self.transform(image=img)
        item= trans_img['image']
        item=item/255.0
        return item

    def __len__(self):
        return len(self.files)

class COCO_Test_Dataset(Dataset):
    def __init__(self, transforms_=None):
        self.transform = transforms_
        self.files = natsorted(
                sorted(glob.glob("/root/autodl-fs/datasets/test2017"+"/*."+"jpg")))

    def __getitem__(self, index):
        while True:
            idx = index % len(self.files)
            img_path = self.files[idx]
            img = cv2.imread(img_path)
            if img.size[0] < 256 or img.size[1] < 256:
                index += 1
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB
            trans_img = self.transform(image=img)
            item= trans_img['image']
            item=item/255.0
            return item

    def __len__(self):
        return len(self.files)

# dataloader
DIV2K_train_cover_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A, mode="train"),
    batch_size=args.single_batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
    persistent_workers=True, # 必须为 True！防止每个 Epoch 重新初始化卡顿
    prefetch_factor=2,        # 预取因子，2 就够了，太多占内存
    drop_last=True
)

DIV2K_train_secret_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A, mode="train"),
    batch_size=args.single_batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
    persistent_workers=True, # 必须为 True！防止每个 Epoch 重新初始化卡顿
    prefetch_factor=2,        # 预取因子，2 就够了，太多占内存
    drop_last=True
)

DIV2K_val_cover_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A_valid, mode="val"),
    batch_size=32,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
    drop_last=True
)

DIV2K_val_secret_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A_valid, mode="val"),
    batch_size=32,
    shuffle=False,
    pin_memory=True,
    num_workers=4,
    drop_last=True
)

DIV2K_test_cover_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A_test, mode="val"),
    batch_size=1,
    shuffle=True,
    pin_memory=True,
    num_workers=1,
    drop_last=True
)

DIV2K_test_secret_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A_test, mode="val"),
    batch_size=1,
    shuffle=True,
    pin_memory=True,
    num_workers=1,
    drop_last=True
)


# COCO_train_cover_loader = DataLoader(
#     COCO_Dataset(transforms_=transform_A, mode="train"),
#     batch_size=args.single_batch_size,
#     shuffle=True,
#     pin_memory=True,
#     num_workers=4,
#     persistent_workers=True, # 必须为 True！防止每个 Epoch 重新初始化卡顿
#     prefetch_factor=2,        # 预取因子，2 就够了，太多占内存
#     drop_last=True
# )
#
# COCO_train_secret_loader = DataLoader(
#     COCO_Dataset(transforms_=transform_A, mode="train"),
#     batch_size=args.single_batch_size,
#     shuffle=True,
#     pin_memory=True,
#     num_workers=4,
#     persistent_workers=True, # 必须为 True！防止每个 Epoch 重新初始化卡顿
#     prefetch_factor=2,        # 预取因子，2 就够了，太多占内存
#     drop_last=True
# )
#
# COCO_val_cover_loader = DataLoader(
#     COCO_Dataset(transforms_=transform_A_valid, mode="val"),
#     batch_size=64,
#     shuffle=True,
#     pin_memory=True,
#     num_workers=4,
#     drop_last=True
# )
#
# COCO_val_secret_loader = DataLoader(
#     COCO_Dataset(transforms_=transform_A_valid, mode="val"),
#     batch_size=64,
#     shuffle=False,
#     pin_memory=True,
#     num_workers=4,
#     drop_last=True
# )
# COCO_test_cover_loader = DataLoader(
#     COCO_Dataset(transforms_=transform_A_test_256, mode="test"),
#     batch_size=1,
#     shuffle=True,
#     pin_memory=True,
#     num_workers=2,
#     drop_last=True
# )
#
# COCO_test_secret_loader = DataLoader(
#     COCO_Dataset(transforms_=transform_A_test_256, mode="test"),
#     batch_size=1,
#     shuffle=True,
#     pin_memory=True,
#     num_workers=2,
#     drop_last=True
# )
