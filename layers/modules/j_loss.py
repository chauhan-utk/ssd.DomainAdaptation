import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import v2 as cfg
from ..box_utils import match, log_sum_exp

class JCLoss(nn.Module):
    """
    A loss function based on Jaccard Similarity index.
    """
    def __init__(self):
        super(JCLoss, self).__init__()
        
    def forward(self, box_a, box_b):
        A = box_a.size(0)
        B = box_b.size(0)
        print("A :",A)
        print("Box 3 :", box_a[2])
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)        
        
        inter = inter[:, :, 0] * inter[:, :, 1]
        area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
        union = area_a + area_b - inter
        result = inter / union  # [A,B]
        #result = result.diag()[result.diag()<0.98] # select places with similarity less than threshold
        result = 1. - result.diag()
        return result.clamp(min=0,max=1).mean()