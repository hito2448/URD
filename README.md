# Unlocking the Potential of Reverse Distillation for Anomaly Detection

[AAAI 2025] PyTorch Implementation of "Unlocking the Potential of Reverse Distillation for Anomaly Detection".
[paper](https://arxiv.org/abs/2412.07579)

## 1. Environment
Create a new conda environment firstly.
```
conda create -n newRD python=3.8
conda activate newRD
pip install -r requirements.txt
```

## 2. Prepare Data
###  MVTec AD Dataset
Download MVTec AD from [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/). 
Unzip the file to `./data/`.
```
|--data
    |-- mvtec_anomaly_detection
        |-- bottle
        |-- cable
        |-- ....
```
###   Describable Textures Dataset
Refer to [DRAEM](https://github.com/VitjanZ/DRAEM), download  Describable Textures dataset from [Describable Textures dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/) for anomaly synthesis. 
Unzip the file to `./data/`.
```
|--data
    |-- dtd
        |-- images
        |-- ....
```

## 3.Train and Test
To get the training and inference results, simply execute the following command.
```
python train.py
```
    
 ## Acknowledgement
Thanks to the codes provided by [Reverse Distillation](https://github.com/hq-deng/RD4AD) which greatly support our work.

## Citation
If you think this work is helpful to you, please consider citing our paper.
```
@article{urd2024,
  title={Unlocking the Potential of Reverse Distillation for Anomaly Detection},
  author={Xinyue Liu and Jianyuan Wang and Biao Leng and Shuo Zhang},
  journal={arXiv preprint arXiv:2412.07579},
  year={2024}
}
```
