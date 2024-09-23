import os
import sys
import pytz
import yaml
import monai
import torch
import ivtmetrics
from torch import nn
from typing import Dict
from objprint import objstr
from easydict import EasyDict
from datetime import datetime
from accelerate import Accelerator
from timm.optim import optim_factory
from monai.utils import ensure_tuple_rep

# src
from src.dataloader import give_dataset
from src.optimizer import give_scheduler
from src.utils import same_seeds, Logger, get_weight_balancing, set_param_in_device, step_params, resume_train_state
# model
from src.models.rendezvous import Rendezvous

# config setting
config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))

# gobel evaluation metrics
mAP = ivtmetrics.Recognition(100)
# mAPi = ivtmetrics.Recognition(6)
# mAPv = ivtmetrics.Recognition(10)
# mAPt = ivtmetrics.Recognition(15)
mAP.reset_global()
# mAPi.reset_global()
# mAPv.reset_global()
# mAPt.reset_global()

def val(model, dataloader, loss_functions, activation, step=0, train=False):
    mAP.reset_global()
    # mAPi.reset_global()
    # mAPv.reset_global()
    # mAPt.reset_global()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    mAP.reset()  
    # mAPv.reset() 
    # mAPt.reset() 
    # mAPi.reset()
    if train == False:
        data_set = 'Val'
    else:
        data_set = 'Train'
    with torch.no_grad():
        model.eval()
        for _, (img, (y1, y2, y3, y4)) in enumerate(dataloader):
            tool, verb, target, triplet = model(img)
            _, logit_i = tool
            _, logit_v = verb
            _, logit_t = target
            logit_ivt  = triplet  
            # loss
            loss_i     = loss_functions['loss_fn_i'](logit_i, y1.float())
            loss_v     = loss_functions['loss_fn_v'](logit_v, y2.float())
            loss_t     = loss_functions['loss_fn_t'](logit_t, y3.float())
            loss_ivt   = loss_functions['loss_fn_ivt'](logit_ivt, y4.float())  
            loss       = (loss_i) + (loss_v) + (loss_t) + loss_ivt
            # all score
            # mAPi.update(y1.float().detach().cpu(), activation(logit_i).detach().cpu()) # Log metrics 
            # mAPv.update(y2.float().detach().cpu(), activation(logit_v).detach().cpu()) # Log metrics 
            # mAPt.update(y3.float().detach().cpu(), activation(logit_t).detach().cpu()) # Log metrics 
            mAP.update(y4.float().detach().cpu(), activation(triplet).detach().cpu())
            # log loss
            if train==False:
                accelerator.log({
                    'Val/Total Loss': float(loss.item()),
                    'Val/loss_i': float(loss_i.item()),
                    'Val/loss_v': float(loss_v.item()),
                    'Val/loss_t': float(loss_t.item()),
                    'Val/loss_ivt': float(loss_ivt.item()),
                    }, step=step)
                step += 1
    mAP.video_end() 
    # mAPv.video_end()
    # mAPt.video_end()
    # mAPi.video_end()
    
    APscore = mAP.compute_video_AP()['mAP']
    
    mAP_i = mAP.compute_video_AP('i', ignore_null=True)
    mAP_v = mAP.compute_video_AP('v', ignore_null=True)
    mAP_t = mAP.compute_video_AP('t', ignore_null=True)
    
    mAP_iv = mAP.compute_video_AP('iv', ignore_null=True)
    mAP_it = mAP.compute_video_AP('it', ignore_null=True)
    mAP_ivt = mAP.compute_video_AP('ivt', ignore_null=True) 
    
    metrics = {
        f'{data_set}/APscore': round(APscore * 100, 3),
        f'{data_set}/I': round(mAP_i["mAP"] * 100, 3),
        f'{data_set}/V': round(mAP_v["mAP"] * 100, 3),
        f'{data_set}/T': round(mAP_t["mAP"] * 100, 3),
        f'{data_set}/IV': round(mAP_iv["mAP"] * 100, 3),
        f'{data_set}/IT': round(mAP_it["mAP"] * 100, 3),
        f'{data_set}/IVT': round(mAP_ivt["mAP"] * 100, 3)
    }
    
    return metrics, step
    
def train_one_epoch(config, model, train_loader, loss_functions, optimizers, schedulers, accelerator, epoch, step):
    # train
    model.train()
    for batch, (img, (y1, y2, y3, y4)) in enumerate(train_loader):
        # output 4 result
        tool, verb, target, triplet = model(img)
        _, logit_i  = tool
        _, logit_v  = verb
        _, logit_t  = target
        logit_ivt   = triplet                
        loss_i      = loss_functions['loss_fn_i'](logit_i, y1.float())
        loss_v      = loss_functions['loss_fn_v'](logit_v, y2.float())
        loss_t      = loss_functions['loss_fn_t'](logit_t, y3.float())
        loss_ivt    = loss_functions['loss_fn_ivt'](logit_ivt, y4.float())  
        loss        = (loss_i) + (loss_v) + (loss_t) + loss_ivt 
        # Backpropagation # optimizer.zero_grad()
        for param in model.parameters():
            param.grad = None
        # lose backward
        accelerator.backward(loss)

        # optimizer.step
        step_params(optimizers)
        # log
        accelerator.log({
            'Train/Total Loss': float(loss.item()),
            'Train/loss_i': float(loss_i.item()),
            'Train/loss_v': float(loss_v.item()),
            'Train/loss_t': float(loss_t.item()),
            'Train/loss_ivt': float(loss_ivt.item()),
        }, step=step)
        step += 1
        accelerator.print(
            f'Epoch [{epoch+1}/{config.trainer.num_epochs}][{batch + 1}/{len(train_loader)}] Losses => total:[{loss.item():.4f}] ivt: [{loss_ivt.item():.4f}] i: [{loss_i.item():.4f}] v: [{loss_v.item():.4f}] t: [{loss_t.item():.4f}]', flush=True)
    # learning rate schedule update
    step_params(schedulers)
    accelerator.print(f'[{epoch+1}/{config.trainer.num_epochs}] Epoch Losses => total:[{loss.item():.4f}] ivt: [{loss_ivt.item():.4f}] i: [{loss_i.item():.4f}] v: [{loss_v.item():.4f}] t: [{loss_t.item():.4f}]', flush=True)    

    if config.trainer.val_training == True:
        metrics, _ = val(model, train_loader, loss_functions, activation, step=0, train=True)
        APscore = metrics['Train/APscore']
        i_score = metrics['Train/I']
        t_score = metrics['Train/T']
        v_score = metrics['Train/V']
        iv_score = metrics['Train/IV']
        it_score = metrics['Train/IT']
        ivt_score = metrics['Train/IVT']
        accelerator.print(f'[{epoch+1}/{config.trainer.num_epochs}] Training Metrics => APscore:[{APscore}] i: [{i_score}] v: [{v_score}] t: [{t_score}] iv: [{iv_score}] iv: [{it_score}] ivt: [{ivt_score}]', flush=True)    
        accelerator.log(metrics, step=epoch)
    
    return step

def val_one_epoch(config, model, val_loader, loss_functions, activation, epoch, step):
    metrics, step = val(model, val_loader, loss_functions, activation, step=step, train=False)
    # ivt_score = metrics['Val/APscore']
    # i_score = metrics['Val/APiscore']
    # t_score = metrics['Val/APtscore']
    # v_score = metrics['Val/APvscore']
    
    APscore = metrics['Val/APscore']
    i_score = metrics['Val/I']
    t_score = metrics['Val/T']
    v_score = metrics['Val/V']
    iv_score = metrics['Val/IV']
    it_score = metrics['Val/IT']
    ivt_score = metrics['Val/IVT']
    
    accelerator.print(f'[{epoch+1}/{config.trainer.num_epochs}] Val Metrics => APscore:[{APscore}] i: [{i_score}] v: [{v_score}] t: [{t_score}] iv: [{iv_score}] it: [{it_score}] ivt: [{ivt_score}] ', flush=True)    
    accelerator.log(metrics, step=epoch)
    return APscore, metrics, step

if __name__ == '__main__':
    same_seeds(50)
    logging_dir = os.getcwd() + '/logs/' + str(datetime.now())
    accelerator = Accelerator(cpu=False, log_with=["tensorboard"], logging_dir=logging_dir)
    Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(config), flush=True)
    
    # load dataset
    train_loader, val_loader, test_loader = give_dataset(config.dataset)
    
    # load model
    model = Rendezvous('resnet18', hr_output=False, use_ln=False).cuda()
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # load optimizer and scheduler
    module_i        = list(set(model.parameters()) - set(model.encoder.cagam.parameters()) - set(model.encoder.bottleneck.parameters()) - set(model.decoder.parameters()))
    module_ivt      = list(set(model.encoder.bottleneck.parameters()).union(set(model.decoder.parameters())))
    module_vt       = model.encoder.cagam.parameters()
    wp_lr           = [lr/config.trainer.power for lr in config.trainer.lr]
    optimizers = {
        'optimizer_i': torch.optim.SGD(module_i, lr=wp_lr[0], weight_decay=config.trainer.weight_decay),
        'optimizer_vt': torch.optim.SGD(module_vt, lr=wp_lr[1], weight_decay=config.trainer.weight_decay),
        'optimizer_ivt': torch.optim.SGD(module_ivt, lr=wp_lr[2], weight_decay=config.trainer.weight_decay)
        }
    schedulers = {
        'scheduler_i': give_scheduler(config, optimizers['optimizer_i'], 0),
        'scheduler_vt': give_scheduler(config, optimizers['optimizer_vt'], 1),
        'scheduler_ivt': give_scheduler(config, optimizers['optimizer_ivt'], 2),
    }
    
    # activation
    activation = nn.Sigmoid()
    
    # load loss
    tool_weight, verb_weight, target_weight = get_weight_balancing(config.dataset)
    loss_functions = {
        'loss_fn_i': nn.BCEWithLogitsLoss(pos_weight=torch.tensor(tool_weight).to(accelerator.device)),
        'loss_fn_v': nn.BCEWithLogitsLoss(pos_weight=torch.tensor(verb_weight).to(accelerator.device)),
        'loss_fn_t': nn.BCEWithLogitsLoss(pos_weight=torch.tensor(target_weight).to(accelerator.device)),
        'loss_fn_ivt': nn.BCEWithLogitsLoss(),
    }
    
    # training setting
    train_step = 0
    val_step = 0
    start_num_epochs = 0
    best_score = -1
    best_metrics = {}
    
    # resume
    if config.trainer.resume:
        model, optimizers, schedulers, start_num_epochs, train_step, val_step, best_score, best_metrics = resume_train_state(model, config.finetune.checkpoint, optimizers, schedulers, accelerator)
        
    # load in accelerator
    optimizers = set_param_in_device(accelerator, optimizers)
    schedulers = set_param_in_device(accelerator, schedulers)
    model, train_loader, val_loader = accelerator.prepare(model, train_loader, val_loader)
    
    for epoch in range(start_num_epochs, config.trainer.num_epochs):
        # train
        train_step = train_one_epoch(config, model, train_loader, loss_functions, optimizers, schedulers, accelerator, epoch, train_step)
        score, metrics, val_step = val_one_epoch(config, model, val_loader, loss_functions, activation, epoch, val_step)
        
        # save best model
        if best_score < score:
            best_score = score
            best_metrics = metrics
            # two types of modeling saving
            accelerator.save_state(output_dir=f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/best/new/")
            torch.save(model.state_dict(), f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/best/new/model.pth")
            torch.save({'epoch': epoch, 'best_score': best_score, 'best_metrics': best_metrics, 'train_step': train_step, 'val_step': val_step},
                    f'{os.getcwd()}/model_store/{config.finetune.checkpoint}/best/epoch.pth.tar')
            
        # print best score
        accelerator.print(f'Now best APscore: {best_score}', flush=True)
        
        # checkout
        accelerator.print('Checkout....')
        accelerator.save_state(output_dir=f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/checkpoint")
        torch.save({'epoch': epoch, 'best_score': best_score, 'best_metrics': best_metrics, 'train_step': train_step, 'val_step': val_step},
                    f'{os.getcwd()}/model_store/{config.finetune.checkpoint}/checkpoint/epoch.pth.tar')
        accelerator.print('Checkout Over!')
        
    accelerator.print(f"dice ivt score: {best_score}")
    accelerator.print(f"other metrics : {best_metrics}")
    sys.exit(1) 