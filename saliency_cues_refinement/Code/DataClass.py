#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3' 

import logging
import time
from os.path import exists, join, split
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

try:
  from modules import batchnormsync
except ImportError:
  pass

import pdb
from tqdm import tqdm

import data_transforms as transforms
from utils import *
from Par_CRF import apply_dcrf_par
from Par_CRF import apply_dcrf_single
from Par_CRF import apply_dcrf
from Par_CRF import save_compute_crf

class BatchData():
    """
    Takes batch as an input. batch is a tuple of len 4, that contains:
        - the input image
        - GT label
        - pseudolabel list
        - image names
    This is the content of one dataloader.
    """
    def __init__(self, batch, active=True):
        assert len(batch)==4
        self.batch = batch
        self.active = active
        self.initialize()

    #initialize self.input, GT_label, pseudolabels, image names
    def initialize(self):
        self.input = self.batch[0]
        self.GT_label = self.batch[1]
        self.pseudolabels = self.batch[2]
        self.names = self.batch[3]
        self.imnames = [(path.split('/')[-1])[:-4] + '.png' for path in self.names]
        self.size = self.GT_label.shape[2]      

    #check if GT labels and pseudolabels have the same dimensions
    def check_dimension(self):
        assert self.GT_label.shape[1]==self.size
        for target in self.pseudolabels:
            assert target.shape == self.GT_label.shape

    #normalize GT label and pseudolabels to range [0,1]
    def normalize_labels(self):
        self.GT_label=self.GT_label.float()/255.0
        for dummy_ind in range(len(self.pseudolabels)):
            self.pseudolabels[dummy_ind]=self.pseudolabels[dummy_ind].float()/255.0

    #create Variables for input and labels. Put them on cuda. Put input to cuda.
    def create_vars_on_cuda(self):
        self.input = self.input.cuda()
        self.input_var = torch.autograd.Variable(self.input).cuda()
        self.GT_label_var = torch.autograd.Variable(self.GT_label).cuda()
        self.pseudolabels_var=[]
        for target in self.pseudolabels:
            self.pseudolabels_var.append(torch.autograd.Variable(target).cuda())
        assert len(self.pseudolabels)==len(self.pseudolabels_var)

    #compute saliecy from model, normalize it and save it as self.sal_pred
    #if DCRF is set, apply deep crf
    #make list with saliency prediction, one entry for each pseudolabel
    def compute_saliency(self, model, DCRF):
        out = model(self.input_var)
        self.out_all = out
        self.output = out[0]
        self.features = out[2]
        self.features_seg = out[3]

        m=torch.nn.Softmax(dim=1)
        self.sal_pred=m(self.output)
        if DCRF:
            self.sal_pred=apply_dcrf(self.sal_pred, self.names, Color=DCRF=='Color' or DCRF=='color')
        else:
            self.sal_pred=self.sal_pred[:, 0, :, :]
        assert self.sal_pred.shape==self.GT_label.shape
        self.sal_pred_list=[]
        for dummy_ind in range(len(self.pseudolabels_var)):
            self.sal_pred_list.append(self.sal_pred)

    #discretize the pseudolabels with soft thresholding
    def discretize_pseudolabels(self, Disc_Thr):
        #Apply soft threshold
        for dummy_ind in range(len(self.sal_pred_list)):
            self.sal_pred_list[dummy_ind]=ThresholdPrediction(self.sal_pred_list[dummy_ind], self.pseudolabels_var[dummy_ind], Disc_Thr)
        #Discretize Targets
        for dummy_ind in range(len(self.pseudolabels_var)):
            self.pseudolabels_var[dummy_ind]=Discretize(self.pseudolabels_var[dummy_ind], Disc_Thr).float()

    #Compute 
    def compute_loss(self, beta=1.0):
        self.loss=0.0
        for dummy_ind in range(len(self.sal_pred_list)):
            self.loss+=F_cont(self.sal_pred_list[dummy_ind], self.pseudolabels_var[dummy_ind], b=beta)
        self.loss/=len(self.sal_pred_list)
