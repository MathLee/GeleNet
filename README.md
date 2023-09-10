# GeleNet
This project provides the code and results for 'Salient Object Detection in Optical Remote Sensing Images Driven by Transformer', IEEE TIP, 2023.

# Network Architecture
   <div align=center>
   <img src="https://github.com/MathLee/GeleNet/blob/main/images/GeleNet.png">
   </div>
   
   
# Requirements
   python 3.8 + pytorch 1.9.0


# Saliency maps
   We provide saliency maps of our GeleNet on ORSSD, EORSSD, and ORSI-4199 datasets in './GeleNet_saliencymap_PVT.zip' (PVT-v2-b2 backbone) and './GeleNet_saliencymap_SwinT.zip' (Swin Transformer backbone).
      
   ![Image](https://github.com/MathLee/GeleNet/blob/main/images/table.png)
   
   
# Training
   We use data_aug.m for data augmentation. 
   
   Download [pvt_v2_b2.pth](https://pan.baidu.com/s/1U6Bsyhu0ynXckU6EnJM35w) (code: sxiq), and put it in './model/'. 
   
   Modify paths of datasets, then run train_GeleNet.py.

Note: Our main model is under './model/GeleNet_models.py' (PVT-v2-b2 backbone)



# Pre-trained model and testing
1. Download the pre-trained models (PVT-v2-b2 backbone) on [ORSSD](https://pan.baidu.com/s/1E6Llbauan4QXfgOvnrcP1w) (code: qga2), [EORSSD](https://pan.baidu.com/s/1dY_9UtDb5GVb9rFyBNDSCA) (code: ahm7), and [ORSI-4199](https://pan.baidu.com/s/1NPdsGBW72vGXgsZxYrJCcA) (code: 5h3u), and put them in './models/'.

2. Modify paths of pre-trained models and datasets.

3. Run test_GeleNet.py.

   
# Evaluation Tool
   You can use the [evaluation tool (MATLAB version)](https://github.com/MathLee/MatlabEvaluationTools) to evaluate the above saliency maps.


# [ORSI-SOD_Summary](https://github.com/MathLee/ORSI-SOD_Summary)
   
# Citation
        @ARTICLE{Li_2023_GeleNet,
                author = {Gongyang Li and Zhen Bai and Zhi Liu and Xinpeng Zhang and Haibin Ling},
                title = {Salient Object Detection in Optical Remote Sensing Images Driven by Transformer},
                journal = {IEEE Transactions on Image Processing},
                volume= {32},
                year={2023},
                doi={10.1109/TIP.2023.3314285},
                }
                
                
If you encounter any problems with the code, want to report bugs, etc.

Please contact me at lllmiemie@163.com or ligongyang@shu.edu.cn.
