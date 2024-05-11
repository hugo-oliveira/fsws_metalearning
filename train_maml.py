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
               learner,
               adaptation_steps,
               loss_name,
               device):
    
    weights = loss_weights(sup_msk, device)
    
    # Adapt the model.
    for step in range(adaptation_steps):
        
        sup_prd = learner(sup_img)
        sup_err = loss_fn(sup_prd, sup_msk, weights, loss_name, device)
        
        learner.adapt(sup_err)
    
    # Evaluate the adapted model.
    qry_prd = learner(qry_img)
    qry_err = loss_fn(qry_prd, qry_msk, weights, loss_name, device)
    
    qry_met = accuracy(qry_msk, qry_prd)
    
    return sup_err, qry_err, qry_met

def main(n_shots=5,             # Number of shots of each task.
         meta_lr=0.002,         # LR of the outer loop.
         fast_lr=0.1,           # LR of the inner loop.
         inner_iters=4,         # Number of iterations in inner loop.
         outer_iters=6000,      # Number of iterations in outer loop.
         adaptation_steps=2,    # Number of adaptation steps at each inner loop.
         seed=42                # Random generator seed.
        ):
    
    # Recovering target dataset and task.
    assert len(sys.argv) >= 5, 'Not enough input parameters. Exiting...'
    
    trg_dataset = sys.argv[1]  # Target dataset for testing.
    trg_task = sys.argv[2]     # Target task for testing.
    network_name = sys.argv[3] # Segmentation network architecture.
    loss_name = sys.argv[4]    # Loss function.
    
    # Setting experiment name
    exp_group_name = 'maml_%s_%s_%s_%s' % (network_name, loss_name, trg_dataset, trg_task)
    
    # Setting result directory.
    result_dir = './experiments/'
    check_mkdir(result_dir)
    
    exp_group_dir = os.path.join(result_dir, exp_group_name)
    check_mkdir(exp_group_dir)
    
    exp_ckpt_dir = os.path.join(result_dir, exp_group_name, 'ckpt')
    check_mkdir(exp_ckpt_dir)
    
    # Setting logging file.
    log_path = os.path.join(exp_group_dir, 'train_maml.log')
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
        model = UNet(1, unet_base, 2, final=True)
        model.to(device)
        
        # Enveloping into MAML wrapper.
        model = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    
    elif 'resnet12' in network_name:
        
        # Create model.
        model = FCNResNet12(1, 2, final=True)
        model.to(device)
        
        # Enveloping into MAML wrapper.
        model = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    
    elif 'efficientlab' in network_name:
        
        # Create model.
        model = EfficientLab(1, 2, final=True)
        model.to(device)
        
        # Enveloping into MAML wrapper.
        model = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    
    elif 'deeplabv3' in network_name:
        
        # Create model.
        model = DeepLabV3(1, 2, final=True)
        model.to(device)
        
        # Enveloping into MAML wrapper.
        model = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    
    # Setting optimizer.
    opt = optim.Adam(model.parameters(), meta_lr)
    
    # Lists of error (loss) and metric (jaccard).
    err_list = []
    met_list = []
    
    # Outer loop.
    iteration = 1
    while iteration <= outer_iters:
        
        for outer_batch in meta_loader:
            
            # Tic.
            tic = time.time()
            
            print('Outer loop %d/%d...' % (iteration, outer_iters))
            logging.info('Outer loop %d/%d...' % (iteration, outer_iters))
            sys.stdout.flush()
            
            # Zeroing optimizer gradients.
            opt.zero_grad()
            
            # Initiating counters.
            meta_qry_err = 0.0
            meta_qry_met = 0.0
            
            # Splitting batch.
            sup_img, sup_msk, sup_dns, qry_img, qry_msk, sparsity_list, sup_names, qry_names = outer_batch
            
            # Inner loop iterations.
            for task in range(inner_iters):
                
                print('  Inner loop %d/%d, task "%s", mode "%s"...' % (task + 1, inner_iters, sparsity_list[task]['task'], sparsity_list[task]['mode']))
                logging.info('  Inner loop %d/%d, task "%s", mode "%s"...' % (task + 1, inner_iters, sparsity_list[task]['task'], sparsity_list[task]['mode']))
                sys.stdout.flush()
                
                # Obtaining inner loop batch.
                inner_sup_img = sup_img[task].to(device) # Support images.
                inner_sup_msk = sup_msk[task].to(device) # Support masks.
                inner_qry_img = qry_img[task].to(device) # Query images.
                inner_qry_msk = qry_msk[task].to(device) # Query masks.
                
                # Compute meta-training loss.
                learner = model.clone()
                sup_err, qry_err, qry_met = fast_adapt(inner_sup_img,
                                                       inner_sup_msk,
                                                       inner_qry_img,
                                                       inner_qry_msk,
                                                       learner,
                                                       adaptation_steps,
                                                       loss_name,
                                                       device)
                
                # Backpropagating.
                qry_err.backward()
                
                # Updating counters.
                meta_qry_err += qry_err.item()
                meta_qry_met += qry_met
            
            # Updating lists.
            meta_qry_err /= inner_iters
            meta_qry_met /= inner_iters
            
            err_list.append(meta_qry_err)
            met_list.append(meta_qry_met)
            
            # Print loss and accuracy.
            print('  Meta Val Err', meta_qry_err)
            print('  Meta Val Jac', meta_qry_met)
            logging.info('  Meta Val Err ' + str(meta_qry_err))
            logging.info('  Meta Val Jac ' + str(meta_qry_met))
            sys.stdout.flush()
            
            # Average the accumulated gradients and take optimizer step.
            for p in model.parameters():
                p.grad.data.mul_(1.0 / inner_iters)
            opt.step()
            
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