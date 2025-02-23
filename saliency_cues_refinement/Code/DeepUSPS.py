#!/usr/bin/env python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5,6' 

import logging
import time
from os.path import exists, join, split
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
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
from DataClass import *


torch.manual_seed(0)

FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)

def validate(val_loader, model, epoch, doc_directory, args, print_freq=10):
        batch_time = AverageMeter()

        DOC=Trackometer(epoch)

        # switch to evaluate mode
        model.eval()
 
        end = time.time()
        for i, (input, GT_label, pseudolabels, name) in tqdm(enumerate(val_loader)):
            #====================================================================================================================
            #       Get Image Names, Check Sizes
            #====================================================================================================================
            size=GT_label.shape[2]
            for target in pseudolabels:
                assert target.shape==GT_label.shape
                
            #Get image name without path
            imname = [(path.split('/')[-1])[:-4] + '.png' for path in name]
            #====================================================================================================================
            #make target float and normalize to range [0,1] for each pixel
            if torch.max(GT_label)!=0:
                GT_label=GT_label.float()/torch.max(GT_label).item()
            else:
                GT_label=GT_label.float()

            input = input.cuda()

            input_var = torch.autograd.Variable(input).cuda()
            GT_label_var = torch.autograd.Variable(GT_label).cuda()

            #====================================================================================================================
            #       Compute Output, normalize it. Optionally apply DCRF
            #====================================================================================================================
            # compute output
            output = model(input_var)[0]

            m=torch.nn.Softmax(dim=1)
            sal_pred=m(output)
            if args.DCRF:
                sal_pred=apply_dcrf(sal_pred, name, Color=args.DCRF=='Color' or args.DCRF=='color')
            else:
                 sal_pred=sal_pred[:, 0, :, :]

            #====================================================================================================================
            #       Update Documentation, Print status in terminal (in respective iterations), save maps (in respective epochs)
            #====================================================================================================================
            DOC.update(sal_pred, GT_label_var, [], [])
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            FreqPrint=len(val_loader)//print_freq
            if FreqPrint<1:
                FreqPrint=1
            if i % (FreqPrint) == 0:
                logger.info('Test: [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'MAE GT {MAE_GT.val:.4f} ({MAE_GT.avg:.4f})\t'
                            'F-Score GT {F_GT.val:.3f} ({F_GT.avg:.3f})'
                            .format(i, len(val_loader), batch_time=batch_time, MAE_GT=DOC.L1_GT, F_GT=DOC.F_GT))

        logger.info('\n\nValidation Epoch {}:\t\tMAE (GT) = {:.1f} %\t\tF-score (GT) = {:.1f} %\n'.format(epoch, DOC.L1_GT.avg*100, DOC.F_GT.avg*100))

        f=open(doc_directory + "loss_val.txt", "a")
        f.write('{}\t{}\t{}\n'.format(epoch, DOC.L1_GT.avg, DOC.F_GT.avg))
        f.close()
 
        return DOC.F_GT.avg, DOC.L1_GT.avg
 
 
 
def train(train_loader, model, optimizer, epoch, doc_directory, args, discretization_threshold, refined_labels_directory=None, iter_size=5,
           print_freq=10, TrainMapsOut=False,mva_preds=None,image2indx=None):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
 
    DOC=Trackometer(epoch)
    if TrainMapsOut:
        DOC_plain = Trackometer(epoch)
        DOC_CRF = Trackometer(epoch)
        DOC_MVA = Trackometer(epoch)
 
    Disc_Thr = discretization_threshold
 
    # switch to train mode
    model.train()
    end = time.time()
 
    output_shape = mva_preds.shape
    raw_preds  = torch.zeros((output_shape[0],2,output_shape[1],output_shape[2]))
    gt_targets =  torch.zeros((mva_preds.shape))
    pseudo_targets =  torch.zeros((mva_preds.shape))

    for i, Data in tqdm(enumerate(train_loader)):
        #====================================================================================================================
        #       Get time, Image Names, Check Sizes
        #       Normalize Labels, create variables and put them on cuda
        #====================================================================================================================
        # measure data loading time
        data_time.update(time.time() - end) 
        #initialize batch data
        batch_data=BatchData(Data, active=True)
        #check dimensions of labels
        batch_data.check_dimension()
        #Make GT label and pseudolabels float and normalize to range [0,1]
        batch_data.normalize_labels()
        #Push input to cuda. Create Variables for input and labels.
        batch_data.create_vars_on_cuda()

        #====================================================================================================================
        #       Compute Output, normalize it. Optionally apply DCRF
        #====================================================================================================================
        #compute saliency prediction, normalize with softmax. Optionally apply Threshold.
        batch_data.compute_saliency(model, False)

        #====================================================================================================================
        #       If TrainMapsOut: Save Training Images (Before Optimizer Step!)
        #====================================================================================================================
        if TrainMapsOut:
            m = torch.nn.Softmax(dim=1)
            sal_pred_raw = m(batch_data.output)
            gt_targets[image2indx(batch_data.names)] = batch_data.GT_label
            pseudo_targets[image2indx(batch_data.names)] = batch_data.pseudolabels[0]
            assert len(batch_data.pseudolabels) == 1, 'Only one map should be refined at a time in order to not lose information'
            raw_preds[image2indx(batch_data.names)] = sal_pred_raw.detach().cpu()

        #====================================================================================================================
        #       Discretize Targets and apply 'soft thresholding' to saliency predictions.
        #====================================================================================================================
        #Discretize all pseudolabels and apply soft thresholing
        batch_data.discretize_pseudolabels(Disc_Thr)

        #=====================================================================================================================
        #       Compute Loss, Gradient and perform optimizer Step.
        #====================================================================================================================
        #compute the loss (with asymmetries and all) and save to batch_active.loss
        batch_data.compute_loss(beta=args.beta)
        loss = batch_data.loss

        #pass iter_size batches before updating grad
        if i%iter_size==0:
            optimizer.zero_grad()
        loss.backward()
        if i%iter_size==iter_size-1:
            optimizer.step()

        #====================================================================================================================
        #       Update Documentation
        #====================================================================================================================
        DOC.update(batch_data.sal_pred, batch_data.GT_label_var, batch_data.sal_pred_list, batch_data.pseudolabels_var)
        #losses is redundant with loss DOC.Loss. Kept for convenience.
        losses.update(loss.data.item(), batch_data.input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        FreqPrint=len(train_loader)//print_freq
        if FreqPrint<1:
            FreqPrint=1
        if i % (FreqPrint) == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'L1 Loss GT {loss_L1_GT.val:.4f} ({loss_L1_GT.avg:.4f})\t'
                         .format(epoch, 
                                 i, 
                                 len(train_loader), 
                                 batch_time=batch_time, 
                                 data_time=data_time, 
                                 loss=losses, 
                                 loss_L1_GT=DOC.L1_GT, 
                                 ))


    if TrainMapsOut:
        assert refined_labels_directory is not None, 'Directory for output of refined Maps needs to be specified'
        #Create Output Directories
        path_train = refined_labels_directory 
        path_plain = join(path_train, 'PlainMaps/')
        path_CRF = join(path_train, 'CRFMaps/')
        path_MVA = join(path_train, 'MVAMaps/')
        for path in  [path_train,path_plain,path_CRF,path_MVA]:
            os.makedirs(path,exist_ok=True)
        name = train_loader.dataset.image_list # the order is kept correctly, 0...2499
 
        save_compute_crf(path_plain, path_CRF, path_MVA,
                           name, gt_targets, pseudo_targets, raw_preds, mva_preds, 
                           image2indx,
                           DOC_plain, DOC_CRF, DOC_MVA, 
                           args)

        assert mva_preds.sum()!=0, 'mva_preds was not updated!?'

        logger.info('\n\n\nTraining Maps Extracted in this epoch {}. Results:\n\nPlain:{}\nCRF:{}\nMVA:{}'\
            .format(epoch, str(DOC_plain), str(DOC_CRF), str(DOC_MVA)))
        
        DOC_plain.write_history(refined_labels_directory + "Results_plain.txt")
        DOC_CRF.write_history(refined_labels_directory + "Results_CRF.txt")
        DOC_MVA.write_history(refined_labels_directory + "Results_MVA.txt")

    else:
        DOC.write_history(doc_directory + "loss_train.txt")

    return losses.avg, mva_preds


def train_round(args, target_dirs, output_dir_it, discretization_threshold, MapsOut = False):
    log_handler = logging.FileHandler(output_dir_it+'/log.txt')
    logger.addHandler(log_handler)
    batch_size = args.batch_size
    num_workers = args.workers
    crop_size = args.crop_size

    iter_size_train=args.iter_size

    f=open(output_dir_it + "params.txt", "w")
    f.write("Parameters:\n\n")
    for k, v in args.__dict__.items():
        f.write("{}:\t\t{}\n".format(k, v))
    f.close()

    single_model = DRNSeg(args.arch, 2, None, pretrained=False)

    #load pretrained model for layers that match in size.
    if args.pretrained:
        print('\n')
        load_dict=torch.load(args.pretrained)
        own_dict=single_model.state_dict()
        for name, param in load_dict.items():
            if name not in own_dict:
                #warnings.warn(' Model could not be loaded ! Thats bad ! ')
                print("####")
                continue
            if own_dict[name].size() != load_dict[name].size():
                print('Size of pretrained model and your model does not match in {} ({} vs. {}). Layer stays initialized randomly.'\
                    .format(name, own_dict[name].size(), load_dict[name].size()))
            else:
                own_dict[name].copy_(param)
        print('\n')


    model = torch.nn.DataParallel(single_model).cuda()

    # Data loading code
    data_dir = args.data_dir
    info = json.load(open(join(data_dir, 'info.json'), 'r'))
    normalize = transforms.Normalize(mean=info['mean'],
                                     std=info['std'])
    t = []
    t.extend([transforms.Resize_Image(crop_size),
              transforms.ToTensor(),
              normalize])
    t_val = t     

    #####
    train_loader = torch.utils.data.DataLoader(
        SegList(args, data_dir, 'train', transforms.Compose(t), 
        image_dir= join(args.root_dir, 'Data/01_img/'), gt_dir= join(args.root_dir, 'Data/02_gt/'), 
        targets = target_dirs, list_dir=args.data_dir, out_name=True),
        batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=False, drop_last=True
     )
 


    optimizer = torch.optim.Adam(single_model.optim_parameters(), lr=args.lr)

    cudnn.benchmark = True
    start_epoch = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    mva_preds,image2indx = init_mva_preds(args,train_loader)
    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(args, optimizer, epoch)
        logger.info('Epoch: [{0}]\tlr {1:.2e}'.format(epoch, lr))

        trainloss, mva_preds = train( train_loader, 
                            model, 
                            optimizer, 
                            epoch, 
                            output_dir_it, 
                            args,
                            discretization_threshold,
                            refined_labels_directory=output_dir_it,
                            iter_size=iter_size_train,  
                            print_freq=6, 
                            TrainMapsOut=MapsOut, 
                            mva_preds=mva_preds,
                            image2indx=image2indx)
        assert torch.isnan(mva_preds.sum(dim=(1,2))).sum().item() == 0, 'images are droped since size of data set is not a multiple of batch size'



        checkpoint_path_latest = output_dir_it + 'checkpoint_{:03d}.pth.tar'.format(epoch + 1)

        if (epoch + 1) % args.checkpoint_freq == 0 or epoch==args.epochs:
            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, checkpoint_path_latest)

    return trainloss

def test(args, eval_data_loader, model, num_classes,
          output_dir='pred', save_vis=False):
    with torch.no_grad():
        model.eval()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()
        hist = np.zeros((num_classes, num_classes))

        DOC = Trackometer(0)

        f = open(output_dir + 'Results.txt', 'w')
        for i, (image, GT_label, pseudolabels, name) in enumerate(eval_data_loader):
            data_time.update(time.time() - end)
            #====================================================================================================================
            #       Get Image Names
            #====================================================================================================================

            #Get image name without path
            imname = [(path.split('/')[-1])[:-4] + '.png' for path in name]
            #====================================================================================================================
            #make target float and normalize to range [0,1] for each pixel
            GT_label=GT_label.float()/torch.max(GT_label).item() 
            for dummy_ind in range(len(pseudolabels)):
                pseudolabels[dummy_ind]=pseudolabels[dummy_ind].float()/255.0

            #pad the image, s.t. width and height are both multiples of 8. 
            #This way, the output will have the same shape as the image. The padded part will be thrown away in the output.
            #Get original width and height
            w0=image.shape[2]
            h0=image.shape[3]
            #Get new width, height, that is a multiple of n=8
            n=8
            dw = -w0%n
            dh = -h0%n
            w1 = w0+dw
            h1 = h0+dh
            #pad on the right the missing width and on the bottom the missing height.
            pad_reflection=nn.ReflectionPad2d((0,dh,0,dw))
            im_new=pad_reflection(image)
            #check if padding went well.
            assert torch.all(torch.eq(image,im_new[:,:,:w0,:h0]))

            image_var = Variable(im_new, requires_grad=False)

            final = model(image_var)[0]
            _, pred = torch.max(final, 1)

            #make continuous prediction, then cast it to unit8
            m=torch.nn.Softmax(dim=1)
            sal_pred=m(final)
            if args.DCRF:
                sal_pred=apply_dcrf(sal_pred[:,:,:w0,:h0], name, Color=(args.DCRF=='Color' or args.DCRF=='color'))
            else:
                sal_pred=sal_pred[:, 0, :w0, :h0]

            assert sal_pred.shape==GT_label.shape

            DOC.update(sal_pred, GT_label.cuda(), [sal_pred], [GT_label.cuda()])

            
            sal_pred = (sal_pred*255).int().cpu().data.numpy()
            GT_label = (GT_label*255).int().cpu().data.numpy()

            if save_vis:
                save_output_images(sal_pred, imname, output_dir)
                save_output_images(GT_label, imname, output_dir, name_suffix='_GT')
            
            batch_time.update(time.time() - end)

            end = time.time()
            if i%50 == 0:
                logger.info('Eval: [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                            'L1 {L1.val:.3f} ({L1.avg:.3f})\t'
                            'F-measure {F.val:.3f} ({F.avg:.3f})\t'
                            .format(i, len(eval_data_loader), batch_time=batch_time, data_time=data_time, L1=DOC.L1_GT, F=DOC.F_GT))

            f.write('{}\t{}\t{}\t{}\n'.format(DOC.L1_GT.val, DOC.F_GT.val, DOC.prec_GT.val, DOC.recall_GT.val))

        f.close()
        print(DOC)

        return DOC

 
def test_saliency(args):
    batch_size = args.batch_size
    num_workers = args.workers
    test_dir = join(args.root_dir, 'Doc/Test/')

    for k, v in args.__dict__.items():
        print(k, ':', v)

    single_model = DRNSeg(args.arch, 2, pretrained_model=None, pretrained=False)
    model = torch.nn.DataParallel(single_model).cuda()

    data_dir = args.data_dir
    info = json.load(open(join(data_dir, 'info.json'), 'r'))
    normalize = transforms.Normalize(mean=info['mean'], std=info['std'])

    dataset = SegList_test(args, data_dir, 'test', transforms.Compose([
        transforms.Resize_Image(args.crop_size),
        transforms.ToTensor(),
        normalize,
    ]), image_dir= join(args.root_dir, 'Data/01_img/'), gt_dir= join(args.root_dir, 'Data/02_gt/'),
    list_dir=args.data_dir, out_name=True)
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=False
    )

    cudnn.benchmark = True

    if not args.resume:
        args.resume = join(args.root_dir + 'Doc/Phase_II_Fusion/checkpoint_200.pth.tar')
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    if not exists(test_dir):
        os.makedirs(test_dir)

    DOC = test(args, test_loader, model, 2, save_vis=True, output_dir=test_dir)

    logger.info('MAE = %f', DOC.L1_GT.avg)
 
    return DOC

 

def test_all_datasets(args):
    #for each dataset, we have a dictionary, that contains
    #   - the name
    #   - the Parameters directory
    #   - name of data (***_names.txt file in Param directory)
    #   - the batch size for testing
    test_dir = join(args.root_dir, 'Doc/Test_all/')
    if not exists(test_dir):
        os.makedirs(test_dir)    

    datasets = []

    MSRAB = ECSSD = DUT = SED2 = THUR = False

    MSRAB = True
    ECSSD = True
    DUT = True
    SED2 = True
    '''
    THUR = True
    '''
    if not args.resume:
        args.resume = join(args.root_dir + 'Doc/Phase_II_Fusion/checkpoint_latest.pth.tar')

    #01_MSRAB
    if MSRAB:
        datasets.append({ \
                'name' : '01_MSRAB', \
                'param_dir' : '/media/bigData/_80_User/Dax/UnsupSD/SD_beta/Data/01_MSRAB/Parameters/', \
                'data_prefix' : 'test', \
                'batch_size' : 1 \
                })
    #02_ECSSD
    if ECSSD:
        datasets.append({ \
                'name' : '02_ECSSD', \
                'param_dir' : '/media/bigData/_80_User/Dax/UnsupSD/SD_beta/Data/02_ECSSD/Parameters/', \
                'data_prefix' : 'all', \
                'batch_size' : 1 \
                })

    #03_DUT
    if DUT:
        datasets.append({ \
                'name' : '03_DUT', \
                'param_dir' : '/media/bigData/_80_User/Dax/UnsupSD/SD_beta/Data/03_DUT/Parameters/', \
                'data_prefix' : 'all', \
                'batch_size' : 1 \
                })

    #04_SED2
    if SED2:
        datasets.append({ \
                'name' : '04_SED2', \
                'param_dir' : '/media/bigData/_80_User/Dax/UnsupSD/SD_beta/Data/04_SED2/Parameters/', \
                'data_prefix' : 'all', \
                'batch_size' : 1 \
                })

    #06_THUR
    if THUR:
        datasets.append({ \
                'name' : '06_THUR', \
                'param_dir' : '/media/bigData/_80_User/Dax/UnsupSD/SD_beta/Data/06_THUR/Parameters/', \
                'data_prefix' : 'GT', \
                'batch_size' : 1 \
                })

    #Iterate through the dictionaries and test each dataset
    for dataset in datasets:
        #set correct arguments
        args.dataset_name = dataset['name']
        args.data_dir = dataset['param_dir']
        args.test_data = dataset['data_prefix']
        args.batch_size = dataset['batch_size']
        DOC = test_saliency(args)
        dataset['Result'] = DOC

    print("\n\n\t\t\tMAE\t\tF\t\tprecision\trecall")
    for dataset in datasets:
        print("{name}: \t\t{DOC.L1_GT.avg:.3f}\t\t{DOC.F_GT.avg:.3f}\t\t{DOC.prec_GT.avg:.3f}\t\t{DOC.recall_GT.avg:.3f}"\
            .format(name=dataset['name'], DOC=dataset['Result']) )


    result_file = join(test_dir, 'Test_Results.txt')
    f = open(result_file, 'a')
    f.write("\t\t\tMAE\t\tF\t\tprecision\trecall\n")
    for dataset in datasets:
        f.write("{name}: \t\t{DOC.L1_GT.avg:.5f}\t\t{DOC.F_GT.avg:.5f}\t\t{DOC.prec_GT.avg:.5f}\t\t{DOC.recall_GT.avg:.5f}\n"\
            .format(name=dataset['name'], DOC=dataset['Result']) )  
    f.close()




def train_unsupervised(args):
    #====================================================================================================================
    #       Phase I: Refinement of Pseodulabels
    #====================================================================================================================.
    learning_rates_refinement = [1e-6, 2e-6, 5e-6]
    args.beta = 1.0
    args.epochs = 25
    args.iter_size = min(1, int(20/args.batch_size))
    num_iterations_refinement = len(learning_rates_refinement)
    doc_directory = join(args.root_dir, 'Doc/')
    refined_labels_directory = join(doc_directory, 'Phase_I_Refined_Maps/')
    os.makedirs(doc_directory, exist_ok=True)
    os.makedirs(refined_labels_directory, exist_ok=True)
    pseudolabels = [
        {'name': 'MC', 'data_directory': join(args.root_dir, 'Data/03_mc/'), 'discretization_threshold': 0.31, \
            'F-score_plain': [71.65], 'MAE_plain': [14.41], 'F-score_mva': [71.65], 'MAE_mva': [14.41]}, 
        {'name': 'HS', 'data_directory': join(args.root_dir, 'Data/04_hs/'), 'discretization_threshold': 0.36, \
            'F-score_plain': [71.29], 'MAE_plain': [16.09], 'F-score_mva': [71.29], 'MAE_mva': [16.09]}, 
        {'name': 'DSR', 'data_directory': join(args.root_dir, 'Data/05_dsr/'), 'discretization_threshold': 0.23, \
            'F-score_plain': [72.27], 'MAE_plain': [12.07], 'F-score_mva': [72.27], 'MAE_mva': [12.07]}, 
        {'name': 'RBD', 'data_directory': join(args.root_dir, 'Data/06_rbd/'), 'discretization_threshold': 0.25, \
            'F-score_plain': [75.08], 'MAE_plain': [11.71], 'F-score_mva': [75.08], 'MAE_mva': [11.71]}
    ]
    target_dirs_refined = []
    for pseudolabel in pseudolabels:
        #directory with input targets
        target_dir = [pseudolabel['data_directory']]
        #directory for output targets
        output_dir = join(refined_labels_directory, pseudolabel['name'] + '/')
        os.makedirs(output_dir, exist_ok=True)
        #discretization threshold for this particular pseudolabel
        discretization_threshold = pseudolabel['discretization_threshold']
        for i in range(num_iterations_refinement):
            args.lr = learning_rates_refinement[i]
            #output directory for current iteration
            output_dir_it = join(output_dir, 'Iteration_' + str(i+1) + '/')
            os.makedirs(output_dir_it, exist_ok=True)
            train_round(args, target_dir, output_dir_it, discretization_threshold, MapsOut = True)
            #after one iteration, discretization threshold does not matter too much
            discretization_threshold = 0.5
            target_dir = [join(output_dir_it, 'MVAMaps/')]
            #get Results
            update_plots(refined_labels_directory, output_dir_it, pseudolabel)
        target_dirs_refined.append(target_dir[0])
    
    #====================================================================================================================
    #       Phase II: Fusion of refine Pseudolabels
    #====================================================================================================================
    exit()
    phase_2_directory = join(doc_directory, 'Phase_II_Fusion/')
    os.makedirs(phase_2_directory, exist_ok=True)
    args.epochs = 200
    args.beta = 4.0
    args.lr = 1e-4
    args.iter_size = min(1, int(100/args.batch_size))
    train_round(args, target_dirs_refined, phase_2_directory, 0.5, MapsOut = False)

    create_phase2_plots(phase_2_directory)





def main():
    args = parse_args()
 
 
    if args.cmd == 'train':
        train_unsupervised(args)

    elif args.cmd == 'test':
        args.dataset_name='01_MSRAB'
        test_saliency(args)

    elif args.cmd == 'test_all':
        test_all_datasets(args)

    
if __name__ == '__main__':
     main()
