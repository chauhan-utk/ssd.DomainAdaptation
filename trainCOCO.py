
# coding: utf-8

# In[2]:


import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torchvision
import argparse
from torch.autograd import Variable
import torch.utils.data as data

#from data import v2, v1, AnnotationTransform, VOCDetection, detection_collate, VOCroot, VOC_CLASSES
from data import v2, v1, detection_collate
from data import MSCOCODetection, COCOAnnotationTransform
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import numpy as np
import time
import sys

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

COMMON_CLASSES = ('person', 'bicycle', 'car',
                  'bus', 'train', 'boat',
                   'bird', 'cat', 'dog',
                   'horse', 'sheep', 'cow',
                   'bottle', 'chair', 'airplane',
                 'dining table', 'potted plant', 'tv',
                 'motorcycle', 'couch')

#hyperparameters

version = "v2"
basenet = "vgg16_reducedfc.pth"
jaccard_threshold=0.5
lr=1e-5
cwd = os.getcwd()
save_folder=cwd + "/MSCOCO14weights/"

voc_root= "/new_data/gpu/utkrsh/coco/" # location of the image root directory

annFile = "/new_data/gpu/utkrsh/coco/annotations/instances_train2014.json"
train_img = "/new_data/gpu/utkrsh/coco/images/train2014/"

cuda=True
resume = "./MSCOCO14weights/ssd300_COCO_12000_load.pth" # saved trained weights
start_iter = 12001

if cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(save_folder):
    os.mkdir(save_folder)

#train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
# train_sets = 'train'
ssd_dim = 300  # only support 300 now
means = (104, 117, 123)  # imagenet mean values
num_classes = len(COMMON_CLASSES) + 1
batch_size = 32
accum_batch_size = 32
iter_size = accum_batch_size / batch_size
max_iter = 120000
weight_decay = 0.0005
stepvalues = (10000, 20000, 40000, 80000, 100000, 120000)
gamma = 0.1
momentum = 0.9


# In[3]:


ssd_net = build_ssd('train', 300, num_classes)
net = ssd_net

net = torch.nn.DataParallel(ssd_net)
cudnn.benchmark = True

if resume:
    print("Resuming training, loading weights from {}...".format(resume))
    ssd_net.load_weights(resume)
else:
    vgg_weights = torch.load(save_folder + basenet)
    print('Loading base network...')
    ssd_net.vgg.load_state_dict(vgg_weights)

if cuda:
    net = net.cuda()

def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

if not resume:
    print('Initializing weights...')
    # initialize newly added layers' weights with xavier method
    ssd_net.extras.apply(weights_init)
    ssd_net.loc.apply(weights_init)
    ssd_net.conf.apply(weights_init)

optimizer = optim.SGD(net.parameters(), lr=lr,
                      momentum=momentum, weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, cuda)

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    global lr
    lr = lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# In[4]:


def my_detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets_1 = []
    imgs = []
    for sample in batch:
        # each sample is the result of one query on the dataset object
        imgs.append(sample[0])
        targets_1.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets_1


# In[ ]:


def train():
    net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0
    loss_prev = None
    save_weights = True
    print_interval = 10
    curr_skip = 0
    max_skip = 2
    
    print('Loading Dataset...')
    
    dataset = MSCOCODetection(image_root=train_img, ann_root=annFile, 
                              transform=SSDAugmentation(ssd_dim, means), 
                              target_transform=COCOAnnotationTransform())
    #dataset = MSCOCODetection(image_root=train_img, ann_root=annFile)
    print("Dataset Loaded!")
    
    epoch_size = len(dataset) // batch_size
    print("Training SSD on", dataset.name)
    step_index = 0
    batch_iterator = None
    data_loader = data.DataLoader(dataset, batch_size, num_workers=0,
                             shuffle=True, collate_fn=my_detection_collate,
                             pin_memory=cuda)
    
    for iteration in range(start_iter, max_iter):
        if(not batch_iterator) or (iteration % epoch_size == 0):
            # create batch_iterator
            batch_iterator = iter(data_loader)
        if iteration in stepvalues:
            step_index += 1
            adjust_learning_rate(optimizer, gamma, step_index)
            # reset epoch loss counters
            epoch += 1
        
        images, targets = next(batch_iterator)
        if cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno, volatile=True) for anno in targets]
        
        # forward pass
        t0 = time.time()
        out = net(images)
        
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        if loss_prev is None:
            loss_prev = loss.data
        else:
            # https://discuss.pytorch.org/t/how-to-use-condition-flow/644/5
            if  (torch.abs(loss_prev - loss.data) < 100000).all():
                loss_prev = loss.data
            else:
                # loss value more than enough deviation
                # skip over the current batch
                if curr_skip < max_skip:
                    curr_skip = curr_skip + 1
                    continue
                else:
                    # save the current model input
                    dump = {}
                    dump['out'] = out
                    dump['targets'] = targets
                    dump['loss_l'] = loss_l
                    dump['loss_c'] = loss_c
                    dump['loss_prev'] = loss_prev
                    dump['loss'] = loss
                    with open("./MSCOCO14weights/fail_dump.pkl","wb") as f:
                        torch.save(dump,f)
                    sys.exit("Loss with NaN values. Check log and dump file")
            
        loss.backward()
        optimizer.step()
        t1 = time.time()
        if iteration % print_interval == 0:
            print("Timer: %.4f sec. " % (t1 - t0))
            print("iter: "+ repr(iteration) + "|| loss_loc: %.4f || loss_conf: %.4f || loss: %.4f || " 
                  % (loss_l.data[0], loss_c.data[0], loss.data[0]), end=' ')
            try:
                with open("./MSCOCO14weights/run.txt","a+") as f:
                    f.write("iter: "+ repr(iteration) + "|| loss_loc: %.4f || loss_conf: %.4f || loss: %.4f \n " 
                            % (loss_l.data[0], loss_c.data[0], loss.data[0]))
            except:
                print("Cannot open log file")
        if iteration % 2000 == 0 and save_weights:
            try:
                print(" Saving state, iter: ", iteration)
                torch.save(ssd_net.state_dict(), "./MSCOCO14weights/ssd300_COCO_" +
                          repr(iteration) + ".pth")
            except IOError:
                with open("./MSCOCO14weights/run.txt","a+") as f:
                    f.write("Some file related error in saving the model state\n")
                print("Some file related eror in saving")
            except:
                with open("./MSCOCO14weights/run.txt","a+") as f:
                    f.write("Some other error while saving the model stats\n")
                print("Some other error while saving")
    torch.save(ssd_net.state_dict(), save_folder+"final_model.pth")


# In[ ]:


train()


# %env CUDA_LAUNCH_BLOCKING=1
# 
# d_iter = iter(data_loader)
# d_img, d_targets = next(d_iter)
# if cuda:
#     d_img = Variable(d_img.cuda())
# else:
#     d_img = Variable(d_img)
# d_out = net(d_img)
# if cuda:
#     d_targets = [Variable(anno.cuda(), volatile=True) for anno in d_targets]
# else:
#     d_targets = [Variable(anno, volatile=True) for anno in d_targets]
# 
# print(d_targets)
# 
# import traceback
# 
# try:
#     d_loss1, d_loss2 = criterion(d_out, d_targets)
# except:
#     traceback.print_exc()

# In[ ]:




