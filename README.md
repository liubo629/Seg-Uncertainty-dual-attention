## Seg_Uncertainty
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

![](https://github.com/layumi/Seg_Uncertainty/blob/master/Visual.jpg)

In this repo, we provide the code for the two papers, i.e., 

- MRNet：[Unsupervised Scene Adaptation with Memory Regularization in vivo](https://arxiv.org/pdf/1912.11164.pdf), IJCAI (2020)

- MRNet+Rectifying: [Rectifying Pseudo Label Learning via Uncertainty Estimation for Domain Adaptive Semantic Segmentation](https://arxiv.org/pdf/2003.03773.pdf), IJCV (2020) [[中文介绍]](https://zhuanlan.zhihu.com/p/130220572)

## Table of contents
* [Prerequisites](#prerequisites)
* [Prepare Data](#prepare-data)
* [Training](#training)
* [Testing](#testing)
* [Trained Model](#trained-model)
* [The Key Code](#the-key-code)
* [Related Works](#related-works)
* [Citation](#citation)

### Prerequisites
- Python 3.6
- GPU Memory >= 11G (e.g., GTX2080Ti or GTX1080Ti)
- Pytorch or [Paddlepaddle](https://www.paddlepaddle.org.cn/)


### Prepare Data
Download [GTA5] and [Cityscapes] to run the basic code.
Alternatively, you could download extra two datasets from [SYNTHIA] and [OxfordRobotCar].

- Download [The GTA5 Dataset]( https://download.visinf.tu-darmstadt.de/data/from_games/ )

- Download [The SYNTHIA Dataset]( http://synthia-dataset.net/download/808/)  SYNTHIA-RAND-CITYSCAPES (CVPR16)

- Download [The Cityscapes Dataset]( https://www.cityscapes-dataset.com/ )

- Download [The Oxford RobotCar Dataset]( http://www.nec-labs.com/~mas/adapt-seg/adapt-seg.html )

The data folder is structured as follows:
```
├── data/
│   ├── Cityscapes/  
|   |   ├── data/
|   |       ├── gtFine/
|   |       ├── leftImg8bit/
│   ├── GTA5/
|   |   ├── images/
|   |   ├── labels/
|   |   ├── ...
│   ├── synthia/ 
|   |   ├── RGB/
|   |   ├── GT/
|   |   ├── Depth/
|   |   ├── ...
│   └── Oxford_Robot_ICCV19
|   |   ├── train/
|   |   ├── ...
```

### Training 
Stage-I:
```bash
python train_ms.py --snapshot-dir ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5  --drop 0.1 --warm-up 5000 --batch-size 2 --learning-rate 2e-4 --crop-size 1024,512 --lambda-seg 0.5  --lambda-adv-target1 0.0002 --lambda-adv-target2 0.001   --lambda-me-target 0  --lambda-kl-target 0.1  --norm-style gn  --class-balance  --only-hard-label 80  --max-value 7  --gpu-ids 0,1  --often-balance  --use-se  
```
GTAWANZHENG:
python train_ms.py --snapshot-dir ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_2  --drop 0.1 --warm-up 5000 --batch-size 2 --learning-rate 2e-4 --crop-size 1024,512 --lambda-seg 0.5  --lambda-adv-target1 0.0002 --lambda-adv-target2 0.001   --lambda-me-target 0  --lambda-kl-target 0.1  --norm-style gn  --class-balance  --only-hard-label 80  --max-value 7  --gpu-ids 0  --often-balance  --use-se  
Deeplabv3+ 3and7
python train_ms.py --snapshot-dir ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_7  --drop 0.1 --warm-up 5000 --batch-size 2 --learning-rate 2e-4 --crop-size 1024,512 --lambda-seg 0.5  --lambda-adv-target1 0.0002 --lambda-adv-target2 0.001   --lambda-me-target 0  --lambda-kl-target 0.1  --norm-style gn  --class-balance  --only-hard-label 80  --max-value 7  --gpu-ids 0  --often-balance  --use-se  

python train_ms.py --snapshot-dir ./snapshots/SE_GN_batchsize2_512x256_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_8  --drop 0.1 --warm-up 5000 --batch-size 2 --learning-rate 2e-4 --crop-size 1024,512 --lambda-seg 0.5  --lambda-adv-target1 0.0002 --lambda-adv-target2 0.001   --lambda-me-target 0  --lambda-kl-target 0.1  --norm-style gn  --class-balance  --only-hard-label 80  --max-value 7  --gpu-ids 0  --often-balance  

#加入danet和ll dropout
python train_ms.py --snapshot-dir ./snapshots/SE_GN_batchsize2_512x256_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_9  --drop 0.1 --warm-up 5000 --batch-size 2 --learning-rate 2e-4 --crop-size 512,256 --lambda-seg 0.5  --lambda-adv-target1 0.0002 --lambda-adv-target2 0.001   --lambda-me-target 0  --lambda-kl-target 0.1  --norm-style gn  --class-balance  --only-hard-label 80  --max-value 7  --gpu-ids 0  --often-balance 

#只有主分类器用了danet
python train_ms.py --snapshot-dir ./snapshots/SE_GN_batchsize2_512x256_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_1  --drop 0.1 --warm-up 5000 --batch-size 4 --learning-rate 2e-4 --crop-size 800,600 --lambda-seg 0.5  --lambda-adv-target1 0.0002 --lambda-adv-target2 0.001   --lambda-me-target 0  --lambda-kl-target 0.1  --norm-style gn  --class-balance  --only-hard-label 80  --max-value 7  --gpu-ids 0  --often-balance  --use-se 



python train_ms.py --snapshot-dir ./snapshots/SE_GN_batchsize2_512x256_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_4  --drop 0.1 --warm-up 5000 --batch-size 2 --learning-rate 2e-4 --crop-size 1024,512 --lambda-seg 0.5  --lambda-adv-target1 0.0002 --lambda-adv-target2 0.001   --lambda-me-target 0  --lambda-kl-target 0.1  --norm-style gn  --class-balance  --only-hard-label 80  --max-value 7  --gpu-ids 0  --often-balance   
#10.20：加入Wasserstein
python train_ms_ws2.py --snapshot-dir ./snapshots/SE_GN_batchsize2_512x256_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_5  --drop 0.1 --warm-up 5000 --batch-size 2 --learning-rate 2e-4 --crop-size 280,256 --lambda-seg 0.5  --lambda-adv-target1 0.0002 --lambda-adv-target2 0.001   --lambda-me-target 0  --lambda-kl-target 0.1  --norm-style gn  --class-balance  --only-hard-label 80  --max-value 7  --gpu-ids 0  --often-balance

#10.22 数据集改为acdc：
python train_ms_acdc.py --snapshot-dir ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_1  --drop 0.1 --warm-up 5000 --batch-size 2 --learning-rate 2e-4 --crop-size 800,512 --lambda-seg 0.5  --lambda-adv-target1 0.0002 --lambda-adv-target2 0.001   --lambda-me-target 0  --lambda-kl-target 0.1  --norm-style gn  --class-balance  --only-hard-label 80  --max-value 7  --gpu-ids 0  --often-balance 
10.25 test:
python evaluate_acdc.py --restore-from ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_1/GTA5_30000.pth#max 33.96
python compute_iou_acdc.py data/acdc/gt result/acdc2SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_1

10.25生成伪标签:
python generate_plabel_acdc.py  --restore-from ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_1/GTA5_30000.pth

#10.25 stage2:///未完成
python train_ft_acdc.py --snapshot-dir ./snapshots/1280x640_restore_ft_GN_batchsize9_512x256_pp_ms_me0_classbalance7_kl0_lr1_drop0.2_seg0.5_BN_80_255_0.8_Noaug --restore-from ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_1/GTA5_30000.pth --drop 0.2 --warm-up 5000 --batch-size 9 --learning-rate 1e-4 --crop-size 512,256 --lambda-seg 0.5 --lambda-adv-target1 0 --lambda-adv-target2 0 --lambda-me-target 0 --lambda-kl-target 0 --norm-style gn --class-balance --only-hard-label 80 --max-value 7 --gpu-ids 0 --often-balance  --use-se  --input-size 1280,640  --train_bn  --autoaug False

#10.25 train——fog 、、、train.txt改为train_fog.txt
python train_ms_acdc.py --snapshot-dir ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_fog  --drop 0.1 --warm-up 5000 --batch-size 2 --learning-rate 2e-4 --crop-size 800,512 --lambda-seg 0.5  --lambda-adv-target1 0.0002 --lambda-adv-target2 0.001   --lambda-me-target 0  --lambda-kl-target 0.1  --norm-style gn  --class-balance  --only-hard-label 80  --max-value 7  --gpu-ids 0  --often-balance

10.26 train_snow train.txt改为train_snow.txt
10.28 test
python evaluate_acdc.py --restore-from ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_snow/GTA5_35000.pth#33.55

#评估cityscapes上训练的权重在acdc上的结果
python evaluate_acdc.py --restore-from ./snapshots/SE_GN_batchsize2_512x256_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_4/GTA5_40000.pth # 40000:46.5

#评估acdc上训练的权重在cityscapes上的结果
python evaluate_cityscapes.py --restore-from ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_1/GTA5_30000.pth
#评估在acdc的4个子数据集上的结果：
python compute_iou_acdc.py data/acdc/gt result/acdc2SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_1

python compute_iou_acdc.py data/acdc/gt result/acdc2SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_snow

python compute_iou_acdc.py data/acdc/gt result/acdc2SE_GN_batchsize2_512x256_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_4

#将/data/user4/lby/Seg-Uncertainty-master/data/Cityscapes/data/leftImg8bit中的train_error中的train_fog改为了train train改为train_normal
source cityscapes_fog target acdc_fog
test target cityscapes：
python evaluate_cityscapes.py --restore-from ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_2/GTA5_25000.pth

10.29 不用selfattention的网络
SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_noattention
python evaluate_cityscapes.py --restore-from ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_noattention/GTA5_30000.pth #43.44
10.30 rain
10.31 night
python train_ms_acdc.py --snapshot-dir ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_night  --drop 0.1 --warm-up 5000 --batch-size 2 --learning-rate 2e-4 --crop-size 800,512 --lambda-seg 0.5  --lambda-adv-target1 0.0002 --lambda-adv-target2 0.001   --lambda-me-target 0  --lambda-kl-target 0.1  --norm-style gn  --class-balance  --only-hard-label 80  --max-value 7  --gpu-ids 0  --often-balance

11.3 加入wgan
python train_ms_wg.py --snapshot-dir ./snapshots/SE_GN_batchsize2_800x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_wg  --drop 0.1 --warm-up 5000 --batch-size 2 --learning-rate 2e-4 --crop-size 800,512 --lambda-seg 0.5  --lambda-adv-target1 0.0002 --lambda-adv-target2 0.001   --lambda-me-target 0  --lambda-kl-target 0.1  --norm-style gn  --class-balance  --only-hard-label 80  --max-value 7  --gpu-ids 0  --often-balance

11.4 test wgan
python evaluate_cityscapes_wg.py --restore-from ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_wg/GTA5_30000.pth #miou=0.06 重新训练


rain test：
python evaluate_acdc.py --restore-from ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_rain/GTA5_25000.pth#25000:31.82

python compute_iou_acdc.py data/acdc/gt result/acdc2SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_rain

night test：
python evaluate_acdc.py --restore-from ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_night/GTA5_25000.pth #25000:32.65

python compute_iou_acdc.py data/acdc/gt result/acdc2SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_night

python evaluate_cityscapes.py --restore-from ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_fog/GTA5_35000.pth

11.10

python evaluate_cityscapes.py --restore-from ./snapshots/SE_GN_batchsize2_800x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_wg/GTA5_30000.pth

11.14
python evaluate_cityscapes.py --restore-from ./snapshots/SE_GN_batchsize2_800x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_wg2/GTA5_25000.pth

11.15 加入sml1
python train_ms_sml1.py --snapshot-dir ./snapshots/SE_GN_batchsize2_800x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_sml1  --drop 0.1 --warm-up 5000 --batch-size 2 --learning-rate 2e-4 --crop-size 800,512 --lambda-seg 0.5  --lambda-adv-target1 0.0002 --lambda-adv-target2 0.001   --lambda-me-target 0  --lambda-kl-target 0.1  --norm-style gn  --class-balance  --only-hard-label 80  --max-value 7  --gpu-ids 0  --often-balance

11.17
python evaluate_cityscapes.py --restore-from ./snapshots/SE_GN_batchsize2_800x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_sml1/GTA5_30000.pth

python train_ft_acdc.py --snapshot-dir ./snapshots/1280x640_restore_ft_GN_batchsize9_512x256_pp_ms_me0_classbalance7_kl0_lr1_drop0.2_seg0.5_BN_80_255_0.8_Noaug --restore-from ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_1/GTA5_30000.pth --drop 0.2 --warm-up 5000 --batch-size 6 --learning-rate 1e-4 --crop-size 512,256 --lambda-seg 0.5 --lambda-adv-target1 0 --lambda-adv-target2 0 --lambda-me-target 0 --lambda-kl-target 0 --norm-style gn --class-balance --only-hard-label 80 --max-value 7 --gpu-ids 0 --often-balance  --use-se  --input-size 1280,640  --train_bn  --autoaug False

11.18
python evaluate_acdc.py --restore-from ./snapshots/1280x640_restore_ft_GN_batchsize9_512x256_pp_ms_me0_classbalance7_kl0_lr1_drop0.2_seg0.5_BN_80_255_0.8_Noaug/GTA5_25000.pth#34.67

fog 45.5
rain 43.86
snow 40.05
night 12.19













#只有主分类器用了danet,主分类器都加了dropout
python train_ms.py --snapshot-dir ./snapshots/SE_GN_batchsize2_512x256_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_5  --drop 0.1 --warm-up 5000 --batch-size 2 --learning-rate 2e-4 --crop-size 800,512 --lambda-seg 0.5  --lambda-adv-target1 0.0002 --lambda-adv-target2 0.001   --lambda-me-target 0  --lambda-kl-target 0.1  --norm-style gn  --class-balance  --only-hard-label 80  --max-value 7  --gpu-ids 0  --often-balance  
test:
python evaluate_cityscapes2.py --restore-from ./snapshots/SE_GN_batchsize2_512x256_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_5/GTA5_25000.pth#45.5

#都用了danet，且step40000
python train_ms.py --snapshot-dir ./snapshots/SE_GN_batchsize2_512x256_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_2  --drop 0.1 --warm-up 5000 --batch-size 4 --learning-rate 2e-4 --crop-size 512,256 --lambda-seg 0.5  --lambda-adv-target1 0.0002 --lambda-adv-target2 0.001   --lambda-me-target 0  --lambda-kl-target 0.1  --norm-style gn  --class-balance  --only-hard-label 80  --max-value 7  --gpu-ids 0  --often-balance  --use-se


#加入ce
python train_ms.py --snapshot-dir ./snapshots/SE_GN_batchsize2_512x256_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_10  --drop 0.1 --warm-up 5000 --batch-size 2 --learning-rate 2e-4 --crop-size 1024,512 --lambda-seg 0.5  --lambda-adv-target1 0.0002 --lambda-adv-target2 0.001   --lambda-me-target 0  --lambda-kl-target 0.1  --norm-style gn  --class-balance  --only-hard-label 80  --max-value 7  --gpu-ids 0  --often-balance
#512,256对照danet
python train_ms.py --snapshot-dir ./snapshots/SE_GN_batchsize2_512x256_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_3  --drop 0.1 --warm-up 5000 --batch-size 2 --learning-rate 2e-4 --crop-size 512,256 --lambda-seg 0.5  --lambda-adv-target1 0.0002 --lambda-adv-target2 0.001   --lambda-me-target 0  --lambda-kl-target 0.1  --norm-style gn  --class-balance  --only-hard-label 80  --max-value 7  --gpu-ids 0  --often-balance

python train_ms.py --snapshot-dir ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_10  --drop 0.1 --warm-up 5000 --batch-size 2 --learning-rate 2e-4 --crop-size 1024,512 --lambda-seg 0.5  --lambda-adv-target1 0.0002 --lambda-adv-target2 0.001   --lambda-me-target 0  --lambda-kl-target 0.1  --norm-style gn  --class-balance  --only-hard-label 80  --max-value 7  --gpu-ids 0  --often-balance

#加入ce和wave
python train_ms.py --snapshot-dir ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_11  --drop 0.1 --warm-up 5000 --batch-size 2 --learning-rate 2e-4 --crop-size 1024,512 --lambda-seg 0.5  --lambda-adv-target1 0.0002 --lambda-adv-target2 0.001   --lambda-me-target 0  --lambda-kl-target 0.1  --norm-style gn  --class-balance  --only-hard-label 80  --max-value 7  --gpu-ids 0  --often-balance

去掉class balance、only hard label often balance use-ce

python train_ms.py --snapshot-dir ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_5  --drop 0.1 --warm-up 5000 --batch-size 2 --learning-rate 2e-4 --crop-size 1024,512 --lambda-seg 0.5  --lambda-adv-target1 0.0002 --lambda-adv-target2 0.001   --lambda-me-target 0  --lambda-kl-target 0.1  --norm-style gn      --gpu-ids 0  --use-se  

qudiaoce
python train_ms.py --snapshot-dir ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_6  --drop 0.1 --warm-up 5000 --batch-size 2 --learning-rate 2e-4 --crop-size 1024,512 --lambda-seg 0.5  --lambda-adv-target1 0.0002 --lambda-adv-target2 0.001   --lambda-me-target 0  --lambda-kl-target 0.1  --norm-style gn  --class-balance  --only-hard-label 80  --max-value 7  --gpu-ids 0  --often-balance  


Generate Pseudo Label:
```bash
python generate_plabel_cityscapes.py  --restore-from ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5/GTA5_25000.pth
python generate_plabel_cityscapes.py  --restore-from ./snapshots/GTA2Cityscapes_single_lsgan/GTA5_110000.pth


python generate_plabel_cityscapes.py  --restore-from ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_10/GTA5_30000.pth
```
```

Stage-II (with recitfying pseudo label):
```bash
python train_ft.py --snapshot-dir ./snapshots/1280x640_restore_ft_GN_batchsize9_512x256_pp_ms_me0_classbalance7_kl0_lr1_drop0.2_seg0.5_BN_80_255_0.8_Noaug --restore-from ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5/GTA5_25000.pth --drop 0.2 --warm-up 5000 --batch-size 9 --learning-rate 1e-4 --crop-size 512,256 --lambda-seg 0.5 --lambda-adv-target1 0 --lambda-adv-target2 0 --lambda-me-target 0 --lambda-kl-target 0 --norm-style gn --class-balance --only-hard-label 80 --max-value 7 --gpu-ids 0,1,2 --often-balance  --use-se  --input-size 1280,640  --train_bn  --autoaug False
#ce /data/Cityscapes/data/pseudov3+/
python train_ft.py --snapshot-dir ./snapshots/1280x640_restore_ft_GN_batchsize9_512x256_pp_ms_me0_classbalance7_kl0_lr1_drop0.2_seg0.5_BN_80_255_0.Noaug_1 --restore-from ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_10/GTA5_30000.pth --drop 0.2 --warm-up 5000 --batch-size 9 --learning-rate 1e-4 --crop-size 512,256 --lambda-seg 0.5 --lambda-adv-target1 0 --lambda-adv-target2 0 --lambda-me-target 0 --lambda-kl-target 0 --norm-style gn --class-balance --only-hard-label 80 --max-value 7 --gpu-ids 0 --often-balance  --use-se  --input-size 1280,640  --train_bn  --autoaug False
```
*** If you want to run the code without rectifying pseudo label, please change [[this line]](https://github.com/layumi/Seg_Uncertainty/blob/master/train_ft.py#L20) to 'from trainer_ms import AD_Trainer', which would apply the conventional pseudo label learning. ***
python train_ft.py --snapshot-dir ./snapshots/1280x640_restore_ft_GN_batchsize9_512x256_pp_ms_me0_classbalance7_kl0_lr1_drop0.2_seg0.5_BN_80_255_0.8_Noaug_2 --restore-from ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5/GTA5_25000.pth --drop 0.2 --warm-up 5000 --batch-size 8 --learning-rate 1e-4 --crop-size 512,256 --lambda-seg 0.5 --lambda-adv-target1 0 --lambda-adv-target2 0 --lambda-me-target 0 --lambda-kl-target 0 --norm-style gn --class-balance --only-hard-label 80 --max-value 7 --gpu-ids 0 --often-balance  --use-se  --input-size 1280,640  --train_bn  --autoaug False

### Testing
```bash
python evaluate_cityscapes.py --restore-from ./snapshots/1280x640_restore_ft_GN_batchsize9_512x256_pp_ms_me0_classbalance7_kl0_lr1_drop0.2_seg0.5_BN_80_255_0.8_Noaug/GTA5_25000.pth
python evaluate_cityscapes.py --restore-from ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5/GTA5_25000.pth
python evaluate_cityscapes.py --restore-from ./snapshots/1280x640_restore_ft_GN_batchsize9_512x256_pp_ms_me0_classbalance7_kl0_lr1_drop0.2_seg0.5_BN_80_255_0.8_Noaug_3/GTA5_80000.pth

python evaluate_cityscapes.py --restore-from ./snapshots/SE_GN_batchsize2_512x256_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_10/GTA5_35000.pth###最好的结果，达到47.2 其实size是1024*512

python evaluate_cityscapes2.py --restore-from ./snapshots/SE_GN_batchsize2_512x256_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_2/GTA5_40000.pth

python evaluate_cityscapes.py --restore-from ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_10/GTA5_35000.pth#30000： 46.65

python evaluate_cityscapes.py --restore-from ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_11/GTA5_35000.pth

python evaluate_cityscapes.py --restore-from ./snapshots/SE_GN_batchsize2_512x256_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_3/GTA5_25000.pth#忘记增大batchsize导致没有收敛

python evaluate_cityscapes.py --restore-from ./snapshots/1280x640_restore_ft_GN_batchsize9_512x256_pp_ms_me0_classbalance7_kl0_lr1_drop0.2_seg0.5_BN_80_255_0.Noaug_1/GTA5_60000.pth #45000:50.89

#单个danet
python evaluate_cityscapes.py --restore-from ./snapshots/SE_GN_batchsize2_512x256_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_4/GTA5_40000.pth # 40000:46.5
测试MC：
python evaluate_cityscapes_MC.py --restore-from ./snapshots/SE_GN_batchsize2_512x256_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_4/GTA5_40000.pth # 40000:47.74
#general pseudo
SAVE_PATH = './data/Cityscapes/data/pseudov3+danet/train'
python generate_plabel_cityscapes.py  --restore-from ./snapshots/SE_GN_batchsize2_512x256_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_4/GTA5_40000.pth
#stage2:
python train_ft.py --snapshot-dir ./snapshots/1280x640_restore_ft_GN_batchsize9_512x256_pp_ms_me0_classbalance7_kl0_lr1_drop0.2_seg0.5_BN_80_255_0.Noaug_3 --restore-from ./snapshots/SE_GN_batchsize2_512x256_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5_4/GTA5_40000.pth --drop 0.2 --warm-up 5000 --batch-size 9 --learning-rate 1e-4 --crop-size 512,256 --lambda-seg 0.5 --lambda-adv-target1 0 --lambda-adv-target2 0 --lambda-me-target 0 --lambda-kl-target 0 --norm-style gn --class-balance --only-hard-label 80 --max-value 7 --gpu-ids 0 --often-balance  --use-se  --input-size 1280,640  --train_bn  --autoaug False

python evaluate_cityscapes.py --restore-from ./snapshots/1280x640_restore_ft_GN_batchsize9_512x256_pp_ms_me0_classbalance7_kl0_lr1_drop0.2_seg0.5_BN_80_255_0.Noaug_3/GTA5_25000.pth #25000:50.94

after stage2测试mc：




python evaluate_cityscapes_MC.py --restore-from ./snapshots/1280x640_restore_ft_GN_batchsize9_512x256_pp_ms_me0_classbalance7_kl0_lr1_drop0.2_seg0.5_BN_80_255_0.Noaug_3/GTA5_25000.pth
```

### Trained Model
The trained model is available at https://drive.google.com/file/d/1smh1sbOutJwhrfK8dk-tNvonc0HLaSsw/view?usp=sharing

- The folder with `SY` in name is for SYNTHIA-to-Cityscapes
- The folder with `RB` in name is for Cityscapes-to-Robot Car

### One Note for SYNTHIA-to-Cityscapes
Note that the evaluation code I provided for SYNTHIA-to-Cityscapes is still average the IoU by divide 19.
Actually, you need to re-calculate the value by divide 16. There are only 16 shared classes for SYNTHIA-to-Cityscapes. 
In this way, the result is same as the value reported in paper.

### The Key Code
Core code is relatively simple, and could be directly applied to other works. 
- Memory in vivo:  https://github.com/layumi/Seg_Uncertainty/blob/master/trainer_ms.py#L232

- Recitfying Pseudo label:  https://github.com/layumi/Seg_Uncertainty/blob/master/trainer_ms_variance.py#L166

### Related Works
We also would like to thank great works as follows:
- https://github.com/wasidennis/AdaptSegNet
- https://github.com/RoyalVane/CLAN
- https://github.com/yzou2/CRST

### Citation
```bibtex
@inproceedings{zheng2019unsupervised,
  title={Unsupervised Scene Adaptation with Memory Regularization in vivo},
  author={Zheng, Zhedong and Yang, Yi},
  booktitle={IJCAI},
  year={2020}
}
@article{zheng2020unsupervised,
  title={Rectifying Pseudo Label Learning via Uncertainty Estimation for Domain Adaptive Semantic Segmentation },
  author={Zheng, Zhedong and Yang, Yi},
  journal={International Journal of Computer Vision (IJCV)},
  doi={10.1007/s11263-020-01395-y},
  year={2020}
}
```

