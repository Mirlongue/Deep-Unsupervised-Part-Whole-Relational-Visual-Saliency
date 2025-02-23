# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math

import numpy as np
import torch.utils.data
from tqdm import tqdm

import data_transforms as transforms
from PIL import Image


class SegList(torch.utils.data.Dataset):
    def __init__(self, pseudolabes, transforms,gt_dir):

        self.transforms = transforms
        #####*
        self.image_names_short=[line.strip() for line in open('name.txt', 'r')]

        self.pseudolabel_list=[]
        for pseudolabe in pseudolabes:
            self.pseudolabel_list.append([pseudolabe+name+'.png' for name in self.image_names_short])

        self.gt_list=[gt_dir +name+'.png' for name in self.image_names_short]



    def __getitem__(self, index):

        data=[]

        data.append(Image.open(self.gt_list[index]))

        pseudo_labels=[]
        for single_label_list in self.pseudolabel_list:
            pseudo_labels.append(Image.open(single_label_list[index]))
        data.append(pseudo_labels)
        data=list(self.transforms(*data))
        data.append(self.image_names_short[index])

        return data

    def __len__(self):
        return len(self.image_names_short)


def save_single_map(sal_pred, path,  file_name_short):
    # save plain maps
    ARR = (sal_pred.numpy() * 255).astype(np.uint8)
    im = Image.fromarray(ARR)
    im.save(path + file_name_short)


#L1_GT_temp=ComputeLoss_Single(torch.nn.L1Loss(reduction='mean'), sal_pred, GT_label)
#L1_GT_temp = ComputeLoss_Single(torch.nn.L1Loss(reduce=True), sal_pred, GT_label)
def ComputeLoss_Single(criterion, prediction, label):
    return criterion(prediction, label)


def inv(a):
    return 1-a

def Discretize(In, a):
    return (In>a).float()

#Prepare pred for thresholding
def ThresholdPrediction(pred, target, Disc_Thr):
    t_up=(target>Disc_Thr).int()
    t_low=(target<Disc_Thr).int()
    a1=2
    a0=1
    #Tensor of shape of pred. Value of -1 means 'entry is original entry from pred', a1-1 means 'value should be 1', a0-1 means 'value should be 0'
    Z=a1*(pred>target).int()*t_up + a0*(pred<target).int()*t_low-1
    return (Z==-1).float()*pred + Z.float().clamp(0,1)





def f_count(t_pred,p_pred):
    p_pred=(p_pred>0.5).float()
    t_pred=(t_pred>0.5).float()
    TP=t_pred*p_pred
    FN=(t_pred)*(1-p_pred)
    FP=(1-t_pred)*(p_pred)

    TP=torch.sum(TP,dim=(0,1))
    FN=torch.sum(FN,dim=(0,1))
    FP=torch.sum(FP,dim=(0,1))


    eps=1e-5
    recall =TP/(TP+FN+eps)
    precision = TP/(TP+FP+eps)
    #print("####")
    #print("precision:")
    #print(precision)
    #print("recall:")
    #print(recall)


    b=0.3
    fscore=(1+b)*precision*recall/(b*precision+recall+eps)
    #print("fscore:")
    #print(fscore)

    return recall


### 计算t_pred的召回率,p_pred精确率
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

    b=3
    fscore=(1+b)*precision*recall/(b*precision+recall+eps)


    up=1-(fscore*fscore)

    up=-math.log(up)+eps



    return up




def computefus_update(a, b, c, d):
    w0 = w1 = 0.25
    w2 = w3 = 0.25

    tmp = w0 * a + w1 * b + w2 * c + w3 * d

    for j in range(3):
        c0 = count_update(tmp, a)
        c1 = count_update(tmp, b)
        c2 = count_update(tmp, c)
        c3 = count_update(tmp, d)


        c0 = torch.tensor(c0)
        c1 = torch.tensor(c1)
        c2 = torch.tensor(c2)
        c3 = torch.tensor(c3)
        sum = torch.exp(c0) + torch.exp(c1) + torch.exp(c2) + torch.exp(c3)

        w0 = torch.exp(c0) / sum
        w1 = torch.exp(c1) / sum
        w2 = torch.exp(c2) / sum
        w3 = torch.exp(c3) / sum

        tmp = w0 * a + w1 * b + w2 * c + w3 * d

    return tmp






def w_g(a):
    eps=1e-5
    up=1-(a*a)
    up=-math.log(up)+eps
    return torch.tensor(up)



#####
def fus():
    pseudolabels = []
    pseudolabels.append("E:/Test_fus/refined_nocrf_3/MC/")
    pseudolabels.append("E:/Test_fus/refined_nocrf_3/HS/")
    pseudolabels.append("E:/Test_fus/refined_nocrf_3/DSR/")
    pseudolabels.append("E:/Test_fus/refined_nocrf_3/RBD/")


    t = []
    t.extend([transforms.Resize_Image(352),
              transforms.ToTensor()])

    #SegList(pseudolabels, transforms.Compose(t))

    fus_loader = torch.utils.data.DataLoader(
        SegList(pseudolabels,transforms.Compose(t),"E:/Test_fus/gt/"),
        batch_size=1,shuffle=False,num_workers=0,
        pin_memory=False,drop_last=True)



    tmp_ans = 0
    for i,Data_all in enumerate(fus_loader):



        gt=Data_all[0]
        gt=gt.float()/255.0


        name=Data_all[2]


        Data=Data_all[1]
        for j in range(len(Data)):
            Data[j]=Data[j].float()/255.0


        for j in range(fus_loader.batch_size):


            t_0=Data[0][j]#Discretize(Data[0][j],0.5) mc
            t_1=Data[1][j]#Discretize(Data[1][j],0.5) hs
            t_2=Data[2][j]#Discretize(Data[2][j],0.5) dsr
            t_3=Data[3][j]#Discretize(Data[3][j],0.5) rbd

            tmp_fus = computefus_update(t_0, t_1, t_2, t_3)




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




            p_4_sum=torch.sum(tmp_pred_4)
            p_3_sum=torch.sum(tmp_pred_3)
            p_2_sum=torch.sum(tmp_pred_2)
            p_1_sum=torch.sum(tmp_pred_1)


            if (p_3_sum/p_1_sum<0.4):      #p_3_sum/p_1_sum<0.45 and
                tmp_ans=tmp_ans+1
                #save_single_map(gt[j], 'E:/Test_fus/tmp/', name[j] + '_GT.png')  #####
                #save_single_map(tmp_fus, 'E:/Test_fus/tmp/', name[j] + '_fus.png')  #####
                #print(name[j])
            else:
                print(name[j])





    print(tmp_ans)















if __name__ == '__main__':

    fus()



