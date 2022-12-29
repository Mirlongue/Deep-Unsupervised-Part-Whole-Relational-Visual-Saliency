#!/usr/bin/env python
# -*- coding: utf-8 -*-
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '6,7,8,9' ####

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


####
'''
    core for fus
'''

def inv(a):
    return 1-a

def count_update(t_pred,p_pred):

    p_pred = ThresholdPrediction(p_pred, t_pred, 0.5)
    t_pred = Discretize(t_pred, 0.5).float()



    TP=t_pred*p_pred
    FN=(t_pred)*(1-p_pred)
    FP=(1-t_pred)*(p_pred)

    TP=torch.sum(TP,dim=(0,1))
    FN=torch.sum(FN,dim=(0,1))
    FP=torch.sum(FP,dim=(0,1))

    eps=1e-5
    recall =TP/(TP+FN+eps)
    precision = TP/(TP+FP+eps)

    b=3#####
    fscore=(1+b)*precision*recall/(b*precision+recall+eps)


    up=1-(fscore*fscore)

    up=up.item()

    up=-math.log(up)+eps

    return up


def computefus_update(a,b,c,d):
    w0 = w1=0.25
    w2 = w3=0.25

    tmp= w0*a+w1*b+w2*c+w3*d

    for j in range(3):
        c0 = count_update(tmp, a)
        c1 = count_update(tmp, b)
        c2 = count_update(tmp, c)
        c3 = count_update(tmp, d)
        '''
        sum = c0 + c1 + c2+c3

        w0 = c0 / sum
        w1 = c1 / sum
        w2 = c2 / sum
        w3 = c3 / sum
        '''

        c0=torch.tensor(c0)
        c1=torch.tensor(c1)
        c2=torch.tensor(c2)
        c3=torch.tensor(c3)
        sum= torch.exp(c0)+torch.exp(c1)+torch.exp(c2)+torch.exp(c3)

        w0=torch.exp(c0)/sum
        w1=torch.exp(c1)/sum
        w2=torch.exp(c2)/sum
        w3=torch.exp(c3)/sum
  

        tmp = w0 * a + w1 * b + w2 * c +w3 *d



    return tmp



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

    ####
    '''
        the update in fus
        and compute loss
    '''
    def compute_loss_update(self, mc_mva,hs_mva,dsr_mva,rbd_mva,batch_size,beta=1.0,epoch=50,Disc_Thr=0.5):
        self.loss=0.0
        self.a=torch.zeros((batch_size, self.size, self.size)).cuda()
        self.b=torch.zeros((batch_size, self.size, self.size)).cuda()
        self.c=torch.zeros((batch_size, self.size, self.size)).cuda()
        self.d=torch.zeros((batch_size, self.size, self.size)).cuda()
        
        #print(epoch)
        #exit()
        mc_mva = mc_mva.cuda()
        hs_mva = hs_mva.cuda()
        dsr_mva = dsr_mva.cuda()
        rbd_mva = rbd_mva.cuda()


        fus = torch.zeros((4,batch_size, self.size, self.size)).cuda()
        for i in range(0,batch_size):

            t_0=mc_mva[i]#Discretize_(mc_mva[i],0.5)
            t_1=hs_mva[i]#Discretize_(hs_mva[i],0.5)
            t_2=dsr_mva[i]#Discretize_(dsr_mva[i],0.5)
            t_3=rbd_mva[i]#Discretize_(rbd_mva[i],0.5)

            t_0_1=t_0*t_1
            t_0_2=t_0*t_2
            t_0_3=t_0*t_3
            t_1_2=t_1*t_2
            t_1_3=t_1*t_3
            t_2_3=t_2*t_3

            t_1_2_3=t_1*t_2*t_3
            t_0_2_3=t_0*t_2*t_3
            t_0_1_3=t_0*t_1*t_3
            t_0_1_2=t_0*t_1*t_2

            tmp_pred_1=inv(inv(t_0)*inv(t_1)*inv(t_2)*inv(t_3))
            tmp_pred_2=inv(inv(t_0_1)*inv(t_0_2)*inv(t_0_3)*inv(t_1_2)*inv(t_1_3)*inv(t_2_3))
            tmp_pred_3=inv(inv(t_1_2_3)*inv(t_0_2_3)*inv(t_0_1_3)*inv(t_0_1_2))
            tmp_pred_4=t_0*t_1*t_2*t_3
            
            tmp_sum_1=torch.sum(tmp_pred_1)
            tmp_sum_2=torch.sum(tmp_pred_2)
            tmp_sum_3=torch.sum(tmp_pred_3)
            tmp_sum_4=torch.sum(tmp_pred_4)



            if((tmp_sum_1-tmp_sum_2)*1.6<tmp_sum_3-tmp_sum_4):
                fus[0][i]=tmp_pred_1
            else:
                fus[0][i]=computefus_update(mc_mva[i],hs_mva[i],dsr_mva[i],rbd_mva[i])

            #fus[1][i]=computefus_update(mc_mva[i],dsr_mva[i],rbd_mva[i])
            #fus[2][i]=computefus_update(mc_mva[i],hs_mva[i],rbd_mva[i])
            #fus[3][i]=computefus_update(mc_mva[i],hs_mva[i],dsr_mva[i])
            #exit()
            if(epoch>=50):
                tmp=self.sal_pred[i].clone().detach()#####
                self.a[i] =  mc_mva[i]*0.95+tmp *0.05
                self.b[i] = hs_mva[i] * 0.95 + tmp * 0.05
                self.c[i] = dsr_mva[i] * 0.95 + tmp * 0.05
                self.d[i] = rbd_mva[i] * 0.95 + tmp* 0.05



        ######
        for i in range(0, 1):#####
            self.sal_pred_list[i] = ThresholdPrediction(self.sal_pred_list[i],fus[i], Disc_Thr)

        for i in range(0, 1):#####
            fus[i] = Discretize(fus[i], Disc_Thr).float()



        for i in range(0,1):
            self.loss+=F_cont(self.sal_pred_list[i], fus[i], b=beta)

        self.loss/=1

