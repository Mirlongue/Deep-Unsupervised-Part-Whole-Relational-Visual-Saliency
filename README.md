# Deep Unsupervised Part-Whole Relational Visual Saliency

[Dataset and pretrained weight, refined saliency cues also saliency map(ECSSD,DUTS-TE,HKU-IS,PASCAL-S,DUT-OMRON).](https://pan.baidu.com/s/1rYU4Lk0w0CwCF--fM_wdWg 
) extraction code:anmh




## Training：
```bash
python Code/DeepUSPS.py fus -r /home/dxh/v2 -s 352 --arch drn_d_105 --batch-size 20
```


## saliency cues refinement：

The code is for refinement cues

We drop the use of CRF(Par_CRF.py  line 95)

[Dataset and pretrained weight, refined saliency cues also.](https://pan.baidu.com/s/1rYU4Lk0w0CwCF--fM_wdWg 
) extraction code:anmh


### Training：
```bash
python Code/DeepUSPS.py train -r /home/dxh/v2 -s 432 --arch drn_d_105 --batch-size 20
```

## weed out:

the code is for weed out negative samples


