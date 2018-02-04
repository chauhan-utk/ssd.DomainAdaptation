from data import v2, v1, AnnotationTransform, VOCDetection, detection_collate, VOCroot, VOC_CLASSES
from utils.augmentations import SSDAugmentation
from data import MSCOCODetection, COCOAnnotationTransform
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data as data
from random import shuffle



class commonDataset(data.Dataset):
    def __init__(self, VOCroot, train_sets, ssd_dim, means, coco_img, annFile ):
        self.name = "VOC07_and_COCO14_train"

        self.vocData = VOCDetection(VOCroot, train_sets,
                               SSDAugmentation(ssd_dim, means),
                               AnnotationTransform())
        self.cocoData = MSCOCODetection(image_root=coco_img,
                                   ann_root=annFile,
                                   transform=SSDAugmentation(ssd_dim, means),
                                   target_transform=COCOAnnotationTransform())
        self.vocLen = len(self.vocData)
        self.cocoLen = len(self.cocoData)
        self.commonElements = [(x,y) for x,y in zip(list(range(self.vocLen)), [0]*self.vocLen)]
        tmp = [(x,y) for x,y in zip(list(range(self.cocoLen)), [1]*self.cocoLen)]
        self.commonElements = self.commonElements + self.commonElements + self.commonElements + self.commonElements + self.commonElements + self.commonElements + tmp
        shuffle(self.commonElements)

    def __len__(self):
        return len(self.commonElements)

    def __getitem__(self, index):
        indx, dataset = self.commonElements[index]
        if dataset == 0:
            img, targets = self.vocData[indx]
            dmn_idx = np.zeros(((int)(targets.size/5),1))
            targets = np.hstack((targets,dmn_idx))
            return img,targets
        else:
            img, targets = self.cocoData[indx]
            dmn_idx = np.ones(((int)(targets.size/5),1))
            targets = np.hstack((targets,dmn_idx))
            return img, targets
