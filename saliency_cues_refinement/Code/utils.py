#!/usr/bin/env python
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3' 

import argparse
import json
import logging
import math
from os.path import exists, join, split
import threading

import time

import numpy as np
import shutil

import sys
from PIL import Image
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import drn as drn
import data_transforms as transforms

try:
    from modules import batchnormsync
except ImportError:
    pass

import pdb
import torchvision.utils
import gc
import pydensecrf.densecrf as dcrf

import matplotlib.pyplot as plt


def Print_GPU_Memory():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass

def Print_GPU_Memory_2():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(reduce(op.mul, obj.size()) if len(obj.size()) > 0 else 0, type(obj), obj.size())
        except:
            pass


FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DRNSeg(nn.Module):
    def __init__(self, model_name, classes, pretrained_model=None,
                 pretrained=True, use_torch_up=False):
        super(DRNSeg, self).__init__()
        model = drn.__dict__.get(model_name)(
            pretrained=pretrained, num_classes=1000)
        pmodel = nn.DataParallel(model)
        if pretrained_model is not None:
            pmodel.load_state_dict(pretrained_model)
        self.base = nn.Sequential(*list(model.children())[:-2])

        self.seg = nn.Conv2d(model.out_dim, classes,
                             kernel_size=1, bias=True)

        self.softmax = nn.Softmax()

        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            up = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
                                    output_padding=0, groups=classes,
                                    bias=False)
            fill_up_weights(up)
            up.weight.requires_grad = False
            self.up = up

    def forward(self, x):
        x = self.base(x)
        z = x.clone()
        x = self.seg(x)
        z2 = x.clone()
        y = self.up(x)
        return y[:,:2], x, z, z2[:,2:]


    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param


class SegList(torch.utils.data.Dataset):
    def __init__(self, args, data_dir, phase, transforms, image_dir=None, gt_dir=None, targets=None, list_dir=None,
                 out_name=False):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.out_name = out_name
        self.phase = phase
        self.transforms = transforms
        self.bbox_list = None


        self.args=args
        self.image_dir = image_dir
        self.gt_dir = gt_dir

        self.image_names_short = None
        self.image_list = None
        self.GTlabel_list = None
        self.pseudolabel_list = None

        self.targets = targets

        self.read_lists()

    def __getitem__(self, index):
        data = [Image.open(join(self.data_dir, self.image_list[index]))]
        if self.GTlabel_list is not None:
            data.append(Image.open(join(self.data_dir, self.GTlabel_list[index])))

        pseudo_labels=[]
        if self.pseudolabel_list is not None:
            for single_label_list in self.pseudolabel_list:
                pseudo_labels.append(Image.open(join(self.data_dir, single_label_list[index])))
        data.append(pseudo_labels)

        data = list(self.transforms(*data))

        if self.out_name:
            data.append(self.image_list[index])
        assert len(data)==4

        return tuple(data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        #Image List: They are all in the same directory image_dir, the name is read from image names, the ending of the name is jpg
        self.image_names_short=[line.strip() for line in open(join(self.list_dir, self.phase + '_names.txt'), 'r')]
        self.image_list = [self.image_dir + name + '.jpg' for name in self.image_names_short]
        self.GTlabel_list = [self.gt_dir + name + '.png' for name in self.image_names_short]
        if self.targets is None:
            self.pseudolabel_list = None
        else:
            self.pseudolabel_list=[]
            for target_dir in self.targets:
                self.pseudolabel_list.append([target_dir + name + '.png' for name in self.image_names_short])
        assert len(self.GTlabel_list)==len(self.image_list)
        if self.pseudolabel_list is not None:
            assert len(self.pseudolabel_list[0])==len(self.image_list)


#Dataset for testing: No pseudolabels required
class SegList_test(torch.utils.data.Dataset):
    def __init__(self, args, data_dir, phase, transforms, image_dir=None, gt_dir=None, list_dir=None,
                 out_name=False):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.out_name = out_name
        self.phase = phase
        self.transforms = transforms
        self.bbox_list = None


        self.args=args
        self.image_dir = image_dir
        self.gt_dir = gt_dir

        self.image_names_short = None
        self.image_list = None
        self.GTlabel_list = None
        self.pseudolabel_list = None

        self.read_lists()

    def __getitem__(self, index):
        data = [Image.open(join(self.data_dir, self.image_list[index]))]
        if self.GTlabel_list is not None:
            data.append(Image.open(join(self.data_dir, self.GTlabel_list[index])))

        pseudo_labels=[]
        if self.pseudolabel_list is not None:
            for single_label_list in self.pseudolabel_list:
                pseudo_labels.append(Image.open(join(self.data_dir, single_label_list[index])))
        data.append(pseudo_labels)

        data = list(self.transforms(*data))

        if self.out_name:
            data.append(self.image_list[index])
        assert len(data)==4

        return tuple(data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        #Image List: They are all in the same directory image_dir, the name is read from image names, the ending of the name is jpg
        self.image_names_short=[line.strip() for line in open(join(self.list_dir, self.phase + '_names.txt'), 'r')]
        self.image_list = [self.image_dir + name + '.jpg' for name in self.image_names_short]
        self.GTlabel_list = [self.gt_dir + name + '.png' for name in self.image_names_short]
        self.pseudolabel_list=([self.GTlabel_list])
        assert len(self.GTlabel_list)==len(self.image_list)
        assert len(self.pseudolabel_list[0])==len(self.image_list)

 
def init_mva_preds(args, data_loader):
     '''
     Inint mva_predictions for given loader-object 
     return: 
         mva_predictions: #Samples , initizialized with 0 
         image2indx: function, which takes an iterable images_queries, reutrn a indices_array in torch.
     ''' 
     #warnings.warn('theses names migth not match properly with the data names')
     images_names = data_loader.dataset.image_list
     n_samples = len ( images_names)
     size = args.crop_size
     mva_preds = torch.zeros((n_samples, size, size))
 
     image_mapping = {image_name:i for image_name, i in zip(images_names, range(n_samples))}
 
     def image2indx(images_queries):
         indices = [image_mapping[name] for name in images_queries]
         indices_array = torch.LongTensor(indices)
         return indices_array
     return mva_preds, image2indx


def apply_dcrf(sal_pred, name, Color=True):
    assert sal_pred.shape[2]==sal_pred.shape[3]
    size=sal_pred.shape[2]
    bs=sal_pred.shape[0]
    sal_pred_dcrf=torch.zeros(bs,size,size).cuda()

    for ind in range(bs):
        b_np=sal_pred[ind,:,:,:].detach().cpu().numpy()
        assert b_np.dtype=='float32'
        b_np_resh=b_np.reshape((2,-1))
        d = dcrf.DenseCRF2D(size, size, 2) 
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
        Map_new=np.array(Q).reshape((2,size,size))
        sal_pred_dcrf[ind]=torch.tensor(Map_new[0]).cuda()
    return sal_pred_dcrf


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def ComputeLoss_Single(criterion, prediction, label):
    return criterion(prediction, label)

#Compute: sum over all i criterion(prediction[i], label[i])
def ComputeLoss_FromList(criterion, prediction, label):
    if len(prediction) == 0:
        return torch.tensor(0, dtype=torch.float)
    loss=0
    for dummy_ind in range(len(prediction)):
        loss+=criterion(prediction[dummy_ind], label[dummy_ind])
    loss/=len(prediction)
    return loss


class Trackometer(object):
    """Computes and stores the average and current values of the losses, F-measure and precision"""
    def __init__(self, epoch):
        self.epoch=epoch
        self.reset()

    def reset(self):
        #on pseudolabels
        self.Loss=AverageMeter()
        self.L1=AverageMeter()
        #on GT
        self.L1_GT=AverageMeter()
        self.F_GT=AverageMeter()
        self.prec_GT=AverageMeter()
        self.recall_GT=AverageMeter()

    def update(self, sal_pred, GT_label, sal_pred_list, pseudolabels):
        if int(torch.__version__[0])>0:
            L1_temp=ComputeLoss_FromList(torch.nn.L1Loss(reduction='mean'), sal_pred_list, pseudolabels)
            L1_GT_temp=ComputeLoss_Single(torch.nn.L1Loss(reduction='mean'), sal_pred, GT_label)
        else:
            L1_temp=ComputeLoss_FromList(torch.nn.L1Loss(reduce=True), sal_pred_list, pseudolabels)
            L1_GT_temp=ComputeLoss_Single(torch.nn.L1Loss(reduce=True), sal_pred, GT_label)
        F_GT_temp,prec_GT_temp,recall_GT_temp=get_F_beta(sal_pred, GT_label, beta_sq=0.3)

        Loss_temp=torch.tensor(0, dtype=torch.float).cuda()#####
        for dummy_ind in range(len(pseudolabels)):
            Loss_temp += F_cont(sal_pred_list[dummy_ind], pseudolabels[dummy_ind], b=1.0)
        Loss_temp /= max(len(pseudolabels), 1)

        self.L1_GT.update(L1_GT_temp.item(), sal_pred.size(0))
        self.F_GT.update(torch.mean(F_GT_temp).item(), sal_pred.size(0))
        self.prec_GT.update(torch.mean(prec_GT_temp).item(), sal_pred.size(0))
        self.recall_GT.update(torch.mean(recall_GT_temp).item(), sal_pred.size(0))

        self.Loss.update(Loss_temp.item(), sal_pred.size(0))
        self.L1.update(L1_temp.item(), sal_pred.size(0))

    def __str__(self):
        return ('\n\nEpoch {:03}\t\t\tLoss\t\tL1\t\tGT: L1\t\tGT: F-measure\tGT: precision\tGT: recall\n'.format(self.epoch) + \
            'Loss\t\t\t\t{:06.4f}\t\t{:06.4f}\t\t{:06.4f}\t\t{:06.4f}\t\t{:06.4f}\t\t{:06.4f}\n' \
            .format(self.Loss.avg,self.L1.avg,self.L1_GT.avg,self.F_GT.avg,self.prec_GT.avg,self.recall_GT.avg))

    def write_history(self, filename):
        f=open(filename, "a")
        f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(self.epoch,self.Loss.avg,self.L1.avg,self.L1_GT.avg,self.F_GT.avg,self.prec_GT.avg,self.recall_GT.avg))
        f.close()


#Quality measure F_beta: Weighted harmonic mean of precision and recall
def get_F_beta(prediction, target, beta_sq=0.3):
    assert len(target.shape)==2 or len(target.shape)==3
    assert target.shape==prediction.shape
    #discretized prediction. Float for multiplication
    pred_disc=(prediction>0.5).float()
    if len(target.shape)==2:
        precision=torch.sum(pred_disc*target).item()/torch.sum(pred_disc)
        recall=torch.sum(pred_disc*target).item()/torch.sum(target)
        return (1+beta_sq)*precision*recall/(beta_sq*precision+recall)
    if len(target.shape)==3:
        #sum for each image all entries of respective tensor
        if int(torch.__version__[0])>0:
            aa=torch.sum(pred_disc*target,(1,2))
            bb=torch.sum(pred_disc,(1,2))
            cc=torch.sum(target,(1,2))
        else:
            aa=torch.sum(torch.sum(pred_disc*target,dim=2), dim=1)
            bb=torch.sum(torch.sum(pred_disc,dim=2), dim=1)
            cc=torch.sum(torch.sum(target,dim=2), dim=1)
        precision=aa/bb
        #if no pixels predicted salient, precision is 1 not nan
        precision[torch.isnan(precision)==1]=1
        recall=aa/cc
        #if no GT pixels, recall is 1 not nan
        recall[torch.isnan(recall)==1]=1
        #return tensor of different F_beta values
        result=(1+beta_sq)*precision*recall/(beta_sq*precision+recall)
        #if precision and recall are both 0, F_beta is zero for this image, non nan
        result[torch.isnan(result)==1]=0
        return result, precision, recall

def Discretize(In, a):
    return (In>a)

#Prepare pred for thresholding
def ThresholdPrediction(pred, target, Disc_Thr):
    t_up=(target>Disc_Thr).int()
    t_low=(target<Disc_Thr).int()
    a1=2
    a0=1
    #Tensor of shape of pred. Value of -1 means 'entry is original entry from pred', a1-1 means 'value should be 1', a0-1 means 'value should be 0'
    Z=a1*(pred>target).int()*t_up + a0*(pred<target).int()*t_low-1
    return (Z==-1).float()*pred + Z.float().clamp(0,1)

def F_cont(sal_pred, Disc_Label, b=1.5):
    assert sal_pred.shape==Disc_Label.shape
    #Get True Positives, False Positives, True Negatives (Continuous!)
    TP=sal_pred*Disc_Label
    FP=sal_pred*(1-Disc_Label)
    TN=(1-sal_pred)*Disc_Label
    #sum up TP,FP, for each image
    if int(torch.__version__[0])>0:
        TP=torch.sum(TP, dim=(1,2))
        FP=torch.sum(FP, dim=(1,2))
        TN=torch.sum(TN, dim=(1,2))
    else: #the above does not work in torch 0.4, which we need for ffi for Deeplab
        TP=torch.sum(torch.sum(TP, dim=2), dim=1)
        FP=torch.sum(torch.sum(FP, dim=2), dim=1)
        TN=torch.sum(torch.sum(TN, dim=2), dim=1)
    eps=1e-5
    prec=TP/(TP+FP+eps)
    recall=TP/(TP+TN+eps)
 
    F=(1+b)*prec*recall/(b*prec+recall+eps)
    Loss=1-F
     
    return torch.mean(Loss)


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr * (1 - epoch / args.epochs) ** 0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_output_images(predictions, filenames, output_dir, name_suffix=''):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    for ind in range(len(filenames)):
        im = Image.fromarray(predictions[ind].astype(np.uint8))
        fn = os.path.join(output_dir, filenames[ind][:-4] + name_suffix + '.png')
        out_dir = split(fn)[0]
        if not exists(out_dir):
            os.makedirs(out_dir)
        im.save(fn)


def create_phase2_plots(directory):
    with open(directory + 'loss_train.txt') as f:
        lines = f.readlines()
        train_loss = [float(line.split('\t')[1])*100 for line in lines]
        fig = plt.figure()
        plt.grid()
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss in %')
        plt.plot(train_loss)
        fig.savefig(directory + 'train_loss.png', dpi=fig.dpi)

    with open(directory + 'loss_val.txt') as f:
        lines = f.readlines()
        val_mae = [float(line.split('\t')[1])*100 for line in lines]
        fig = plt.figure()
        plt.grid()
        plt.title('Validation Error (MAE) on Ground Truth')
        plt.xlabel('Epoch')
        plt.ylabel('MAE in %')
        plt.plot(val_mae)
        fig.savefig(directory + 'val_mae.png', dpi=fig.dpi)


def update_plots(refined_labels_directory, output_dir_it, pseudolabel):            
    with open(output_dir_it + 'Results_plain.txt', 'r') as file_plain:
        lines_plain = file_plain.readlines()
        results_plain = [float(el) for el in lines_plain[-1].strip('\n').split('\t')]
    with open(output_dir_it + 'Results_MVA.txt', 'r') as file_mva:
        lines_mva = file_mva.readlines()
        results_mva = [float(el) for el in lines_mva[-1].strip('\n').split('\t')]
    pseudolabel['MAE_plain'].append(results_plain[3]*100)
    pseudolabel['F-score_plain'].append(results_plain[4]*100)
    pseudolabel['MAE_mva'].append(results_mva[3]*100)
    pseudolabel['F-score_mva'].append(results_mva[4]*100)

    #plot results
    fig = plt.figure()
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('MAE in %')
    plt.xticks(np.arange(0, len(pseudolabel['F-score_plain'])))
    plt.title('Mean average error of raw maps')
    plt.plot(pseudolabel['MAE_plain'])
    fig.savefig(refined_labels_directory + pseudolabel['name'] + '_raw_mae.png', dpi=fig.dpi)

    fig = plt.figure()
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('MAE in %')
    plt.xticks(np.arange(0, len(pseudolabel['F-score_plain'])))
    plt.title('Mean average error of MVA maps')
    plt.plot(pseudolabel['MAE_mva'])
    fig.savefig(refined_labels_directory + pseudolabel['name'] + '_mva_mae.png', dpi=fig.dpi)

    fig = plt.figure()
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('F-score in %')
    plt.xticks(np.arange(0, len(pseudolabel['F-score_plain'])))
    plt.title('F-score of raw maps')
    plt.plot(pseudolabel['F-score_plain'])
    fig.savefig(refined_labels_directory + pseudolabel['name'] + '_raw_F.png', dpi=fig.dpi)

    fig = plt.figure()
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('F-score in %')
    plt.xticks(np.arange(0, len(pseudolabel['F-score_plain'])))
    plt.title('F-score of MVA maps')
    plt.plot(pseudolabel['F-score_mva'])
    fig.savefig(refined_labels_directory + pseudolabel['name'] + '_mva_F.png', dpi=fig.dpi)


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('cmd', choices=['train', 'test', 'test_all'])
    parser.add_argument('-r', '--root-dir', default=None)
    parser.add_argument('--beta', type=float, default=1.0,
                        help='beta_sq for continuous F-measure')
    parser.add_argument('--checkpoint_freq', type=int, default=25, help='Frequency for saving checkpoints for model')
    parser.add_argument('-s', '--crop-size', default=432, type=int)
    parser.add_argument('--arch', type=str, default='drn_d_105')
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--iter_size', type=int, default=1, metavar='ItS',
                        help='input iteration size for training, over how many batches gradient is accumulated')
    parser.add_argument('--DCRF', type=str, default=None,
                        help='If set, Dense CRF is applied in validation and test. If set to Color, it also uses rgb image for the DCRF')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained',
                        default='', type=str, metavar='PATH',
                        help='use pre-trained model')
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--bn-sync', action='store_true')
    args = parser.parse_args()

    args.data_dir = join(args.root_dir, 'Parameters/')


    if len(args.pretrained)==0:
        if args.arch == 'drn_d_22':
            args.pretrained = join(args.root_dir, 'Pretrained_Models/drn_pretraining/drn_d_22_cityscapes.pth')
        elif args.arch == 'drn_d_38':
            args.pretrained = join(args.root_dir, 'Pretrained_Models/drn_pretraining/drn_d_38_cityscapes.pth')
        elif args.arch == 'drn_d_105':
            args.pretrained = join(args.root_dir, 'Pretrained_Models/drn_pretraining/drn-d-105_ms_cityscapes.pth')

    if args.cmd == 'test':
        args.DCRF = 'Color'

    print(' '.join(sys.argv))
    print(args)

    if args.bn_sync:
        drn.BatchNorm = batchnormsync.BatchNormSync

    return args