
# coding: utf-8

# In[8]:


'''
MSCOCO14 dataset classes
Made using voc0712.py as reference.
'''


# In[1]:


import os
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import json


# In[2]:


#path to coco PythonAPI
# also contain MSCOCO14 dataset
pycocoapi = "/new_data/gpu/utkrsh/coco/PythonAPI/"


# In[3]:


import sys
sys.path.append(pycocoapi)


# In[4]:


from pycocotools.coco import COCO


# In[5]:


COMMON_CLASSES = ('person', 'bicycle', 'car',
                  'bus', 'train', 'boat',
                   'bird', 'cat', 'dog',
                   'horse', 'sheep', 'cow',
                   'bottle', 'chair', 'airplane',
                 'dining table', 'potted plant', 'tv',
                 'motorcycle', 'couch')


# In[6]:


class MSCOCODetection(data.Dataset):
    def __init__(self,image_root,ann_root,transform=None,target_transform=None,
                dataset_name="COCO2014"):
        self.image_root = image_root
        self.ann_root = ann_root
        self.transform = transform
        self.target_transform=target_transform
        self.name=dataset_name
        self.COMMON_CLASSES = ('person', 'bicycle', 'car',
                  'bus', 'train', 'boat',
                   'bird', 'cat', 'dog',
                   'horse', 'sheep', 'cow',
                   'bottle', 'chair', 'airplane',
                 'dining table', 'potted plant', 'tv',
                 'motorcycle', 'couch')
        self.coco = COCO(ann_root)
        self.catIds = self.coco.getCatIds(catNms=self.COMMON_CLASSES)
        self.targetLabels = dict(zip(self.catIds,range(len(self.catIds))))
        self.imgIdList = list() #list of images with given categories
        for i in self.catIds:
            self.imgIdList += self.coco.catToImgs[i]
        self.imgIdList = list(set(self.imgIdList))

    def __getitem__(self, index):
        ''' insert return statement '''
        im, gt, _, _ = self.pull_item(index)
        return im, gt

    def __len__(self):
        ''' return the number of examples '''
        return len(self.imgIdList)

    def pull_item(self,index):
        imgObj = self.coco.loadImgs(self.imgIdList[index])[0] #return list
            #containining dictionary object
        img = cv2.imread(self.image_root+imgObj['file_name']) #numpy array
        height, width, channels = img.shape
        # get all annotations for the current image
        # returns a list containing dictionary objects
        target = self.coco.loadAnns(self.coco.getAnnIds(imgIds=imgObj['id'],
                                    catIds=self.catIds, iscrowd=None))
        # filter out annotations that belong to common classes only
        # list of dictionary objects that contain annotations for the common classes

        #target = [c for c in anno if c['category_id'] in self.catIds]

        if self.target_transform is not None:
            # target will now be a list of lists containing bounding boxes
            # [[xmin, ymin, xmax, ymax, label_ind], ... ]
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            # bbox co-ordinations are the first 4 values
            # class label is the last value for every annotation
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # convert labels in class range
            tmp = [self.targetLabels[i] for i in labels]
            labels = np.array(tmp)
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        imgObj = self.coco.loadImgs(self.imgIdList[index])[0]
        return cv2.imread(self.image_root+imgObj['file_name'],
                          cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        imgObj = self.coco.loadImgs(self.imgIdList[index])[0]
        # returns dictionary object
        annId = self.coco.getAnnIds(imgIds=imgObj['id'],
                                    catIds=self.catIds, iscrowd=None)
        ann = self.coco.loadAnns(annId) #return list of annotations
        gt = self.target_transform(anno, 1, 1)
        return imgObj['id'], gt


# In[7]:


class COCOAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        return

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : list of dictionary objects

        Returns:
            a list containing lists of bounding boxes  [bbox coords, class index]
        """
        res = []
        for i in target:
            bbox = i['bbox']

            #bbox format is [xmin, ymin, width, height]
            bbox[0] = bbox[0] + 1 # 1-index to 0-index
            bbox[1] = bbox[1] + 1
            bbox[2] = bbox[2] + bbox[0]
            bbox[3] = bbox[3] + bbox[1]
            bbox[0] /= width
            bbox[2] /= width
            bbox[1] /= height
            bbox[3] /= height
            bbox.append(i['category_id'])
            res += [bbox]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]
