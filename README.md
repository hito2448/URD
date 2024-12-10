# Unlocking the Potential of Reverse Distillation for Anomaly Detection

PyTorch Implementation of "Unlocking the Potential of Reverse Distillation for Anomaly Detection".
[paper](https://arxiv.org/abs/)

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

```
