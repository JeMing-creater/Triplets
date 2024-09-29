# Experiment environment of triplet datasets

## setting
#### all params of this experiment environment are included in \config.yml, user can change connection params for your detail environment, especially the `config.dataset.data_dir`.

#### this code only require the version of accelerate <= 0.18.0.

## get project
```
git clone https://github.com/JeMing-creater/Triplets.git
```

## training
### two types of training:

#### single GPU training: 
```
python3 main.py
```

#### mutliple GPUs training:

```
sh run.sh
```
#### tips: user can change setting of mutliple GPUs in run.sh. For now, mutliple GPUs training utilizes 2 GPUs as training device, the first two devices are used by default.

## tensorboard
```
tensorboard --logdir=/logs
```

## huggingface 
If you encounter pre-trained model parameter connection network errors in China, please change the mirror proxy: 
### Linux：
```
pip install huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
```
