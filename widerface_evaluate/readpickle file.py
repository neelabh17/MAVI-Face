"""
WiderFace evaluation code
author: wondervictor
mail: tianhengcheng@gmail.com
copyright@wondervictor
"""

import os
import tqdm
import pickle
import argparse
import numpy as np
from scipy.io import loadmat
from bbox import bbox_overlaps
from IPython import embed



""" gt dir: (wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat, wider_hard_val.mat)"""

gt_mat = loadmat('wider_face_val.mat')
hard_mat = loadmat( 'wider_hard_val.mat')
medium_mat = loadmat( 'wider_medium_val.mat')
easy_mat = loadmat( 'wider_easy_val.mat')

facebox_list = gt_mat['face_bbx_list']
event_list = gt_mat['event_list']
file_list = gt_mat['file_list']

hard_gt_list = hard_mat['gt_list']
medium_gt_list = medium_mat['gt_list']
easy_gt_list = easy_mat['gt_list']

print(facebox_list[0].shape)

# return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list