import numpy as np
from PIL import Image
import imgaug.augmenters as iaa
import torchvision
import torch
import random

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

def get_cifar10(root, n_labeled,
                 transform_train=None, transform_val=None,
                 download=True):

    base_dataset = torchvision.datasets.CIFAR10(root, train=True, download=download)
    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(base_dataset.targets, int(n_labeled/10))

    train_labeled_dataset = CIFAR10_labeled(root, train_labeled_idxs, train=True, transform=transform_train)
    train_unlabeled_dataset = CIFAR10_unlabeled(root, train_unlabeled_idxs, train=True, transform=TransformTwice(transform_train))
    val_dataset = CIFAR10_labeled(root, val_idxs, train=True, transform=transform_val, download=True)
    test_dataset = CIFAR10_labeled(root, train=False, transform=transform_val, download=True)

    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #Val: {len(val_idxs)}")
    print(len(test_dataset))
    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset
    

def train_val_split(labels, n_labeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-500])
        val_idxs.extend(idxs[-500:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs

cifar10_mean = (0.4914, 0.4822, 0.4465) 
cifar10_std = (0.2471, 0.2435, 0.2616) 

def normalize(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target]) 

############# Concept taken from an implementation of MixMtach. I have implemented it using that concept
class RandomPadandCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, x):
        x = np.pad(x, [(0, 0), (4, 4), (4, 4)], mode='reflect')

        height, width = x.shape[1:]
        new_height, new_width = self.output_size

        bb_height = np.random.randint(0, height - new_height)
        bb_width = np.random.randint(0, width - new_width)

        img = x[:, bb_height: bb_height + new_height, bb_width: bb_width + new_width]

        return img

####################### This Class is implemented by me to implement random horizontal flip on images
class RandomFlip(object):
    """Flip randomly the image.
    """
    def __call__(self, x):
        for i in range(len(x)):
            n = random.uniform(0.1,0.5)
            x[i] = iaa.Fliplr(p=0.4).augment_image(x[i])
            

        return x.copy()

###################### This class is implemented by me to perform gaussian blur on images.
class GaussianNoise(object):
    def __call__(self, x):
        for i in range(len(x)):
            x[i] = x[i].filter(ImageFilter.GaussianBlur)
        return x
    
class ToTensor(object):
    """Transform the image to tensor.
    """
    def __call__(self, x):
        x = torch.from_numpy(x)
        return x
    
############# The class implementation was taken from an internet resource for MixMatch implementation. I liked their idea of implementing this class which inherits from "torchvision.datasets.CIFAR10" and this implementation made things easier

class CIFAR10_labeled(torchvision.datasets.CIFAR10):

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_labeled, self).__init__(root, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.data = transpose(normalize(self.data))
    

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
       
        
        
        return img, target
    
######### The concept of this class implementation was taken from an internet resource for MixMatch implementation.
class CIFAR10_unlabeled(CIFAR10_labeled):

    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_unlabeled, self).__init__(root, indexs, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        self.targets = np.array([-1 for i in range(len(self.targets))])
        