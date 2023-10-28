from torchvision import transforms as tf
import torchvision.transforms
import torch


rootDir = "/data/data/ped.bmi"
mean, std= torch.tensor([0.4359, 0.3436, 0.2514]), torch.tensor([0.2183, 0.2509, 0.2213])


def getTransform (split):
    cropPadLR = 64
    cropPadBU = 64

    transforms_train = torchvision.transforms.Compose([
        tf.ToPILImage(),
        tf.RandomRotation(degrees=(-5, 5)),
        tf.ColorJitter(brightness=0.1, contrast=0.10),
        tf.ToTensor(),
        tf.Normalize(mean, std),
        tf.RandomErasing(p=0.5, scale=(0.01, 0.15), ratio=(0.3, 3.3), value=(0,0,0))
    ])

    transforms_test = transforms_val = torchvision.transforms.Compose([
        tf.ToPILImage(),
        #tf.Resize((imgSize, imgSize)),
        tf.ToTensor(),
        tf.Normalize(mean, std)
    ])

    if split == "train":
        return transforms_train
    elif split == "val" or split == "test" or split == "valid":
        return transforms_val
    pass

#
