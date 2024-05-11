import os
import sys
import copy
import time
import random
import logging
import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional
from torch.utils import data

from sklearn import metrics

from utils.utils import *
from utils.experiments import *
from utils.losses import *
from models import *
from data.meta_dataset import *

import learn2learn as l2l
from learn2learn.data.transforms import (NWays,
                                         KShots,
                                         LoadData,
                                         RemapLabels,
                                         ConsecutiveLabels)

def fast_adapt(sup_img,
               sup_msk,
               qry_img,
               qry_msk,
               model,
               loss_name,
               device):
    
    # Adapt the model.
    weights = loss_weights(sup_msk, device)
    
    # Computing embeddings.
    sup_emb = model(sup_img)
    qry_emb = model(qry_img)
    
    # Permuting embedding dimensions.
    sup_emb = sup_emb.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
    qry_emb = qry_emb.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
    
    # Linearizing support and query masks.
    lin_sup_msk = sup_msk.view(-1) # [B, H, W] -> [B * H * W]
    lin_qry_msk = qry_msk.view(-1) # [B, H, W] -> [B * H * W]
    
    # Linearizing batch and spacial dimensions of support and query.
    sup_emb = torch.reshape(sup_emb, (-1, sup_emb.shape[3])) # [B, H, W, C] -> [B * H * W, C]
    qry_emb = torch.reshape(qry_emb, (-1, qry_emb.shape[3])) # [B, H, W, C] -> [B * H * W, C]
    
    ########################################################################
    max_size = 16384
    if lin_sup_msk.shape[0] > max_size:
        sup_perm = torch.randperm(lin_sup_msk.size(0))[:max_size]
        lin_sup_msk = lin_sup_msk[sup_perm]
        sup_emb = sup_emb[sup_perm]
        
    if lin_qry_msk.shape[0] > max_size:
        qry_perm = torch.randperm(lin_qry_msk.size(0))[:max_size]
        lin_qry_msk = lin_qry_msk[qry_perm]
        qry_emb = qry_emb[qry_perm]
    ########################################################################
    
    # Acquiring negative and positive samples from support.
    neg_sup = sup_emb[lin_sup_msk == 0]
    pos_sup = sup_emb[lin_sup_msk == 1]
    
    # Computing prototypes for negative and positive classes on support.
    proto_neg_sup = neg_sup.mean(dim=0).unsqueeze(0)
    proto_pos_sup = pos_sup.mean(dim=0).unsqueeze(0)
    
    # Computing query logits according to support prototypes.
    logits_qry = cosine_logits(qry_emb, proto_neg_sup, proto_pos_sup)
    
    lin_qry_prd = logits_qry.argmax(1)
    
    # Acquiring negative and positive predictions from query.
    neg_qry = qry_emb[lin_qry_prd == 0]
    pos_qry = qry_emb[lin_qry_prd == 1]
    
    # Computing prototypes for negative and positive classes on support.
    proto_neg_qry = neg_qry.mean(dim=0).unsqueeze(0)
    proto_pos_qry = pos_qry.mean(dim=0).unsqueeze(0)
    
    # Computing query logits according to support prototypes.
    logits_sup = cosine_logits(sup_emb, proto_neg_qry, proto_pos_qry)
    
    # Computing supervised loss on query predictions.
    qry_err = loss_fn(logits_qry, lin_qry_msk, weights, loss_name, device) + loss_fn(logits_sup, lin_sup_msk, weights, loss_name, device)
    
    qry_met = accuracy(lin_qry_msk, logits_qry)
    
    return qry_err, qry_met

def main(n_shots=5,             # Number of shots of each task.
         meta_lr=0.002,         # LR of the outer loop.
         inner_iters=4,         # Number of iterations in inner loop.
         outer_iters=6000,      # Number of iterations in outer loop.
         seed=42                # Random generator seed.
        ):
    
    # Recovering target dataset and task.
    assert len(sys.argv) >= 5, 'Not enough input parameters. Exiting...'
    
    trg_dataset = sys.argv[1]  # Target dataset for testing.
    trg_task = sys.argv[2]     # Target task for testing.
    network_name = sys.argv[3] # Segmentation network architecture.
    loss_name = sys.argv[4]    # Loss function.
    
    pretrained = None
    if len(sys.argv) >=6:
        pretrained = sys.argv[5]  # Optional parameter with the path to a renset18 or resnet50 model pretrained with SSL 

    if 'resnet50' in network_name or 'resnet18' in network_name:
        
        meta_lr = 0.1
    
    # Setting experiment name
    exp_group_name = 'panet_%s_%s_%s_%s' % (network_name, loss_name, trg_dataset, trg_task)
    
    # Setting result directory.
    result_dir = './experiments/'
    check_mkdir(result_dir)
    
    exp_group_dir = os.path.join(result_dir, exp_group_name)
    check_mkdir(exp_group_dir)
    
    exp_ckpt_dir = os.path.join(result_dir, exp_group_name, 'ckpt')
    check_mkdir(exp_ckpt_dir)
    
    # Setting logging file.
    log_path = os.path.join(exp_group_dir, 'train_panet.log')
    logging.basicConfig(filename=log_path, format='%(message)s', level=logging.INFO)
    
    # Setting random seeds.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda:0')
    
    # Parameters.
    root = '../Datasets/'
    fold = 0
    if 'resnet50' in network_name or 'resnet18' in network_name:
        normalization = 'fixed'
    else:
        normalization = 'minmax'
#     normalization = 'z-score'
    resize_to = (140,140)
    crop_to = (128,128)
    debug = False
    verbose = False
    unet_base = 32
    
    batch_size = 3
    num_workers = 6
    
    print_freq = 100
    test_freq = 6000
    
    # Loss selection.
    if loss_name == 'sce':
        loss_name = 'SCE'
    elif loss_name == 'dice':
        loss_name = 'Dice'
    elif loss_name == 'sce+dice':
        loss_name = 'SCE+Dice'
    elif loss_name == 'focal':
        loss_name = 'Focal'
    
    shots_list = [1, 5, 10, 20]
    
    # Setting dataset.
    meta_set = MetaDataset(root=root,
                           fold=fold,
                           trg_dataset=trg_dataset,
                           resize_to=resize_to,
                           crop_to=crop_to,
                           normalization=normalization,
                           num_shots=n_shots,
                           debug=debug,
                           verbose=verbose)
    
    # Setting dataloader.
    meta_loader = data.DataLoader(meta_set,
                                  batch_size=inner_iters,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  collate_fn=collate_meta)
    
    if 'unet' in network_name:
        
        # Create model.
        model = UNet(1, unet_base, 2, final=False)
        model.to(device)
    
    elif 'efficientlab' in network_name:
        
        # Create model.
        model = EfficientLab(1, 2, final=False)
        model.to(device)
    
    elif 'deeplabv3' in network_name:
        
        # Create model.
        model = DeepLabV3(1, 2, final=False)
        model.to(device)
    
    elif 'resnet12' in network_name:
        
        # Create model.
        model = FCNResNet12(1, 2, final=False)
        model.to(device)
    
    elif 'resnet18' in network_name:
        
        # Create model.
        model = FCNResNet18(1, 2, final=False, pretrained=pretrained)
        model.to(device)
    
    elif 'resnet50' in network_name:
        
        # Create core model.
        model = FCNResNet50(1, 2, final=False, pretrained=pretrained)
        model.to(device)
    
    # Setting optimizer and LR scheduler.
    opt = optim.Adam(model.parameters(), meta_lr)
    
    scheduler = optim.lr_scheduler.StepLR(opt, outer_iters // 3, gamma=0.5)
    
    # Lists of error (loss) and metric (jaccard).
    err_list = []
    met_list = []
    
    # Outer loop.
    iteration = 1
    while iteration <= outer_iters:
        
        for outer_batch in meta_loader:
                
            torch.cuda.empty_cache()
            
            # Tic.
            tic = time.time()
            
            print('Outer loop %d/%d...' % (iteration, outer_iters))
            logging.info('Outer loop %d/%d...' % (iteration, outer_iters))
            sys.stdout.flush()
            
            # Initiating counters.
            meta_qry_err = 0.0
            meta_qry_met = 0.0
            
            # Splitting batch.
            sup_img, sup_msk, sup_dns, qry_img, qry_msk, sparsity_list, sup_names, qry_names = outer_batch
            
            # Inner loop iterations.
            inner_count = 0
            for task in range(inner_iters):
                
                print('  Inner loop %d/%d, task "%s", mode "%s"...' % (task + 1, inner_iters, sparsity_list[task]['task'], sparsity_list[task]['mode']))
                logging.info('  Inner loop %d/%d, task "%s", mode "%s"...' % (task + 1, inner_iters, sparsity_list[task]['task'], sparsity_list[task]['mode']))
                sys.stdout.flush()
                
                # Zeroing optimizer gradients.
                opt.zero_grad()
                
                # Obtaining inner loop batch.
                inner_sup_img = sup_img[task].to(device) # Support images.
                inner_sup_msk = sup_msk[task].to(device) # Support masks.
                inner_qry_img = qry_img[task].to(device) # Query images.
                inner_qry_msk = qry_msk[task].to(device) # Query masks.
                
                # Compute meta-training loss.
                qry_err, qry_met = fast_adapt(inner_sup_img,
                                              inner_sup_msk,
                                              inner_qry_img,
                                              inner_qry_msk,
                                              model,
                                              loss_name,
                                              device)
                
                if not torch.any(torch.isnan(qry_err)):
                    
                    # Backpropagating.
                    qry_err.backward()
                    
                    # Taking optimization step.
                    opt.step()
                    
                    # Updating counters.
                    meta_qry_err += qry_err.detach().item()
                    meta_qry_met += qry_met
                    
                    inner_count += 1
            
            # Updating lists.
            meta_qry_err /= inner_count
            meta_qry_met /= inner_count
            
            err_list.append(meta_qry_err)
            met_list.append(meta_qry_met)
            
            # Print loss and accuracy.
            print('  Meta Val Err', meta_qry_err)
            print('  Meta Val Jac', meta_qry_met)
            logging.info('  Meta Val Err ' + str(meta_qry_err))
            logging.info('  Meta Val Jac ' + str(meta_qry_met))
            sys.stdout.flush()
            
            # Scheduler step.
            scheduler.step()
            
            # Toc.
            toc = time.time()
            print('  Duration %.1f seconds' % (toc - tic))
            logging.info('  Duration %.1f seconds' % (toc - tic))
            sys.stdout.flush()
            
            # Meta-testing.
            if iteration % test_freq == 0:
                
                # Saving model and optimizer.
                model_ckpt_path = os.path.join(exp_ckpt_dir, 'model.pth')
                optim_ckpt_path = os.path.join(exp_ckpt_dir, 'optim.pth')
                
                torch.save(model, model_ckpt_path)
                torch.save(opt, optim_ckpt_path)
            
            # Updating iteration.
            iteration = iteration + 1
            
            if iteration > outer_iters:
                break

if __name__ == '__main__':
    main()