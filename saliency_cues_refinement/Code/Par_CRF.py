#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torchvision
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from tqdm import tqdm

import os


import numpy as np
from PIL import Image
import torch
from torch import nn

try:
    from modules import batchnormsync
except ImportError:
    pass

import pydensecrf.densecrf as dcrf
import warnings
import time 

import pdb

def save_single_map(sal_pred, path,  file_name_short, GT_label):
    # save plain maps
    ARR = (sal_pred.numpy() * 255).astype(np.uint8)
    im = Image.fromarray(ARR)
    im.save(path + file_name_short)
    # if GT not yet in folder, also save GT
    GT_name = path + file_name_short[:-4] + '_0GT.png'
    if not os.path.isfile(GT_name):
        ARR = (GT_label.numpy() * 255).astype(np.uint8)
        im = Image.fromarray(ARR)
        im.save(GT_name)
    return

def apply_dcrf_single(sal_pred,Color,name):
    assert sal_pred.shape[1]==sal_pred.shape[2]
    size=sal_pred.shape[2]
    assert sal_pred.dtype == 'float32'
    b_np_resh = sal_pred.reshape((2,-1))
    d = dcrf.DenseCRF2D(size, size, 2) 
    with np.errstate(divide='ignore', invalid='ignore'):
        d.setUnaryEnergy(-np.log(b_np_resh))

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).

    if Color: 
        img=np.array(Image.open(name).resize((size,size)))
        #take care of black white images:
        if len(img.shape) == 2:
            img=np.stack((img, img, img), axis=2)
        assert len(img.shape) == 3
        assert img.shape[2] == 3
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(5)
    Map_new = np.array(Q).reshape((2,size,size))
    sal_pred_dcrf = torch.tensor(Map_new[0])
    return sal_pred_dcrf


class CRFDataset(torchvision.datasets.DatasetFolder):

    def __init__(self, sal_preds, images_names=None, Color=True,
                    args = None,
                    crf_on_mva = True,
                 ):
        self.samples = list(zip(sal_preds,images_names))
        self.Color = Color
        self.args = args
        self.crf_on_mva  = crf_on_mva 

    def _save_data(self, sal_pred_CRF,c_sal_pred_MVA, index):
        sal_pred_plain,_, path_MVA,  name,  GT_label, path_CRF, path_plain,trainmap_suffix = self.args
        file_name = name[index]
        gt_label = GT_label[index]
        file_name_short = os.path.basename(file_name)[:-4] + trainmap_suffix + '.png'

        save_single_map(sal_pred_CRF, path_CRF,  file_name_short, gt_label)
        save_single_map(sal_pred_plain[index], path_plain,  file_name_short, gt_label)
        save_single_map(c_sal_pred_MVA, path_MVA,  file_name_short, gt_label) #MVA Off

    def __getitem__(self, index):
        alpha = 0.7
        sal_pred_MVA = self.args[1]
        Color = self.Color
        sal_pred, name = self.samples[index]


        sal_pred_dcrf = torch.tensor(sal_pred[0]) ##### apply_dcrf_single(sal_pred,Color,name)
        #First update with training CRF 
        c_sal_pred_MVA = sal_pred_MVA[index]
        c_sal_pred_MVA = alpha*c_sal_pred_MVA +(1-alpha) * sal_pred_dcrf

        #MVA-CRF
        stacked_c_sal_pred_MVA = np.stack([c_sal_pred_MVA, (1-c_sal_pred_MVA)],axis=0)

        if self.crf_on_mva:
            c_sal_pred_MVA_dcrf = apply_dcrf_single(stacked_c_sal_pred_MVA,Color,name)
        else:
            c_sal_pred_MVA_dcrf = c_sal_pred_MVA

        self._save_data(sal_pred_dcrf,c_sal_pred_MVA_dcrf, index)

        return sal_pred_dcrf,c_sal_pred_MVA, c_sal_pred_MVA_dcrf, index





def init_crf_loader(sal_preds, images_names, args, Color=True, crf_on_mva=True):
    assert sal_preds.shape[2] == sal_preds.shape[3]
    size = sal_preds.shape[2]
    #Batch Size
    sal_preds = sal_preds.detach().cpu().numpy()

    batch_size = 24*5

    crf_dataset = CRFDataset(sal_preds,
                           images_names=images_names,
                           Color=Color,
                           args=args,
                           crf_on_mva = crf_on_mva,
                           )
    batch_sampler = BatchSampler(
                        SequentialSampler(crf_dataset),
                        batch_size=batch_size,
                        drop_last=False,
                        )
    #####
    crf_loader = torch.utils.data.DataLoader(
         crf_dataset,
         batch_sampler=batch_sampler,
         shuffle=False,
         num_workers=0,
         pin_memory=False,
         drop_last=False)
    return crf_loader

def save_compute_crf(path_plain, path_CRF, path_MVA, 
                    images_names, GT_label, Pseudo_label, sal_pred_raw, sal_pred_MVA, 
                    image2indx,
                    DOC_plain, DOC_CRF, DOC_MVA, 
                    args):

    sal_pred_plain = sal_pred_raw[:, 0, :, :]
    assert sal_pred_plain.shape==GT_label.shape

    args = [sal_pred_plain,sal_pred_MVA, path_MVA,  images_names, GT_label, path_CRF, path_plain, '']

    crf_loader = init_crf_loader(sal_pred_raw, images_names, args, Color=True, crf_on_mva=False)
    n_samples   = sal_pred_raw.shape[0]
    size = sal_pred_raw.shape[2]


    #Computation & update the doc 
    sal_pred_CRF = torch.zeros(n_samples,size,size)
    for c_sal_pred_CRF, c_sal_pred_MVA, c_sal_pred_MVA_dcrf, indx in tqdm(crf_loader,'Computing CRFs on all data points'):
        c_sal_pred_plain = sal_pred_plain[indx].cuda()
        c_GT_label = GT_label[indx].cuda()
        c_Pseudo_label = Pseudo_label[indx].cuda()

        sal_pred_MVA[indx] = c_sal_pred_MVA

        c_sal_pred_CRF = c_sal_pred_CRF.cuda()
        c_sal_pred_CRF[torch.isnan(c_sal_pred_CRF)] = 0
        c_sal_pred_MVA_dcrf = c_sal_pred_MVA_dcrf.cuda()
        c_sal_pred_MVA_dcrf[torch.isnan(c_sal_pred_MVA_dcrf)] = 0

        assert c_sal_pred_CRF.shape==c_GT_label.shape
        if int(torch.__version__[0])>0:
            DOC_plain.update(c_sal_pred_plain, c_GT_label, [c_sal_pred_plain], [c_Pseudo_label])
            DOC_CRF.update(c_sal_pred_CRF, c_GT_label, [c_sal_pred_CRF], [c_Pseudo_label])
            DOC_MVA.update(c_sal_pred_MVA_dcrf, c_GT_label, [c_sal_pred_MVA_dcrf], [c_Pseudo_label])
        else:
            DOC_plain.update(c_sal_pred_plain, c_GT_label, [c_sal_pred_plain], [c_Pseudo_label])
            DOC_CRF.update(c_sal_pred_CRF, c_GT_label, [c_sal_pred_CRF], [c_Pseudo_label])
            DOC_MVA.update(c_sal_pred_MVA_dcrf, c_GT_label, [c_sal_pred_MVA_dcrf], [c_Pseudo_label])



#####################################################
#####################################################
#Depricated Code
#@DeprecationWarning
def apply_dcrf(sal_pred, name, Color=True):
    w=sal_pred.shape[2]
    h=sal_pred.shape[3]
    
    size=sal_pred.shape[2]
    bs=sal_pred.shape[0]
    sal_pred_dcrf=torch.zeros(bs,w,h)

    for ind in range(bs):
        b_np=sal_pred[ind,:,:,:].detach().cpu().numpy()
        assert b_np.dtype=='float32'
        b_np_resh=b_np.reshape((2,-1))
        #Mind the order of h and w!
        d = dcrf.DenseCRF2D(h, w, 2) 
        d.setUnaryEnergy(-np.log(b_np_resh))

        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).        
        if Color: 
            img=np.array(Image.open(name[ind]).resize((size,size)))
            #take care of black white images:
            if len(img.shape)==2:
                img=np.stack((img, img, img), axis=2)
            assert len(img.shape)==3
            assert img.shape[2]==3
            d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

        Q = d.inference(5)
        Map_new=np.array(Q).reshape((2,w,h))
        assert b_np.shape==Map_new.shape
        sal_pred_dcrf[ind]=torch.tensor(Map_new[0])
    sal_pred_dcrf  = sal_pred_dcrf.cuda()
    return sal_pred_dcrf

@DeprecationWarning
def apply_dcrf_par(sal_preds, images_names, Color=True):

    assert sal_preds.shape[2] == sal_preds.shape[3]
    size = sal_preds.shape[2]
    bs = sal_preds.shape[0]
    sal_preds = sal_preds.detach().cpu().numpy()
    sal_preds_dcrf = torch.zeros(bs,size,size)

    crf_dataset = CRFDataset(sal_preds,
                           images_names=images_names,
                           Color=Color)
    batch_sampler = BatchSampler(
                        SequentialSampler(crf_dataset),
                        200,
                        drop_last=False,
                        )
    crf_loader = torch.utils.data.DataLoader(
         crf_dataset,
         batch_sampler=batch_sampler,
         shuffle=False,
         num_workers= 0,
         pin_memory=False,
         drop_last=False)
    
    for output_sal_pred_dcrf, indx in tqdm(crf_loader):
        sal_preds_dcrf[indx] = output_sal_pred_dcrf

    return sal_preds_dcrf