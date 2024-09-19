# Experiment environment of triplet datasets

## setting
#### all params of this experiment environment are included in \config.yml, user can change connection params for your detail environment.

#### this code only require the version of accelerate <= 0.18.0.


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


