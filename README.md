# UWF-Net
This is the official code of the paper:

**Automatic Ultra-Wide-Field Fundus Image Enhancement for Improved Clinical Diagnosis**

## Train
### Dataset preparation
1. The dataset including FDUWI-1, FDUWI-2 subset and quality assessment dataset can be accessed by sending a request to 22210240015@m.fudan.edu.cn with detailed reasons for usage. 
2. Organize dataset within the following manner:
   ```
   |--datadir
       |--train
           |--A (low quality image_dir)
              photo1.jpg
              photo1.jpg
              ...
           |--B ( quality image_dir)
              photo1.jpg
              photo1.jpg
              ...
   ```
### Pre-trained model preparation
Prepare your pre-trained disease classification model and Change the model path of it in `Line 200` of `models.py`

### Train your own model
Run `train.py` with
   ```
   cd ./UWF-Net
   python train.py --dataroot "HERE/IS/YOUR/DATADIR/" \
   --save_dir "MODEL/SAVEDIR/"
  ```

## Test
### Model preparation
#### FIQA
1. See [here](https://github.com/hzfu/EyeQ) for model downloading.
2. Change the path of FIQA model in `Line 22` of `fiqa.py`
#### UWFQA
1. Download UWFQA model [here]( https://pan.baidu.com/s/16zcfU3H7qFxA3XPGv7GDaQ) with code mcx9
2. Change the path of UWFQA model in `Line 34` of `uwfqa.py`

### Get FIQA and UWFQA score
Run `test.sh` with 
```
bash test.sh
```
