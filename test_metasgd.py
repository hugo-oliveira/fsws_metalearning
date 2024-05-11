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

def fast_tune(sup_img,
              sup_msk,
              qry_loader,
              model,
              adaptation_steps,
              loss_name,
              device):
    
    # Time counters.
    adapt_time = 0.0
    test_time = 0.0
    
    # Adapt the model.
    weights = loss_weights(sup_msk, device)
    
    # Tic.
    tic = time.time()
    
    for step in range(adaptation_steps):
        
        sup_prd = model(sup_img)
        sup_err = loss_fn(sup_prd, sup_msk, weights, loss_name, device)
        
        model.adapt(sup_err)
    
    # Toc.
    toc = time.time()
    adapt_time = (toc - tic)
    
    # Evaluate the adapted model.
    qry_err_list = []
    qry_met_list = []
    
    # Setting model to evaluation mode.
    model.eval()
    with torch.no_grad():
        
        # Iterating over query batches.
        for batch in qry_loader:
            
            qry_img, qry_msk, qry_names = batch
            
            qry_img = qry_img.to(device)
            qry_msk = qry_msk.to(device)
            
            tic = time.time() # Tic.
            
            qry_prd = model(qry_img)
            
            toc = time.time() # Toc.
            test_time += (toc - tic)
            
            qry_err = loss_fn(qry_prd, qry_msk, weights, loss_name, device)
            qry_met = accuracy(qry_msk, qry_prd)
            
            qry_err_list.append(qry_err.item())
            qry_met_list.append(qry_met)
            
        qry_err_np = np.asarray(qry_err_list)
        qry_met_np = np.asarray(qry_met_list)
        
        print('    Error: %.4f +/- %.4f' % (qry_err_np.mean(), qry_err_np.std()))
        print('    Metric: %.4f +/- %.4f' % (qry_met_np.mean(), qry_met_np.std()))
        logging.info('    Error: %.4f +/- %.4f' % (qry_err_np.mean(), qry_err_np.std()))
        logging.info('    Metric: %.4f +/- %.4f' % (qry_met_np.mean(), qry_met_np.std()))
        sys.stdout.flush()
        
        print('    Adaptation Duration %.4f seconds' % (adapt_time))
        print('    Test Duration %.4f seconds' % (test_time))
        logging.info('    Adaptation Duration %.4f seconds' % (adapt_time))
        logging.info('    Test Duration %.4f seconds' % (test_time))
        sys.stdout.flush()
    
    # Reverting model to training mode.
    model.train()
    
    return qry_err_list, qry_met_list

def meta_test(model, adaptation_steps, loss_name, device, exp_group_dir, exp, sparsity_mode):
    
    sup_list, qry_loader = exp
    
    print('Testing sparsity mode "%s"...' % (sparsity_mode))
    logging.info('Testing sparsity mode "%s"...' % (sparsity_mode))
    
    # Iterating over support datasets with differing n_shots and sparsities.
    for sup in sup_list:
            
        print('  %d-shot' % (sup['shots']), ' sparsity ', sup['sparsity'])
        logging.info('  %d-shot' % (sup['shots']) + ' sparsity ' + str(sup['sparsity']))
        
        # Setting result directory for current experiment.
        if sparsity_mode == 'dense':
            exp_name = '%s_%d-shot' % (sparsity_mode, sup['shots'])
        else:
            exp_name = '%s_%d-shot_%s' % (sparsity_mode, sup['shots'], str(sup['sparsity']).replace(' ', '_'))
        
        # Presetting experiment dir.
        exp_dir = os.path.join(exp_group_dir, exp_name)
        check_mkdir(exp_dir)
        
        # Recovering support dataset.
        sup_img, sup_msk, sup_dns, sup_names = sup['sup_dataset'][0]
        
        # Tuning and saving outputs.
        err_list, met_list = fast_tune(sup_img.to(device),
                                       sup_msk.to(device),
                                       qry_loader,
                                       model.clone(),
                                       adaptation_steps,
                                       loss_name,
                                       device)

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
    
    global is_parallel
    is_parallel = len(sys.argv) == 6 and sys.argv[5] == '--parallel'
    
    # Setting experiment name
    exp_group_name = 'metasgd_%s_%s_%s_%s' % (network_name, loss_name, trg_dataset, trg_task)
    
    # Setting result directory.
    result_dir = './experiments/'
    exp_group_dir = os.path.join(result_dir, exp_group_name)
    exp_ckpt_dir = os.path.join(result_dir, exp_group_name, 'ckpt')
    
    # Setting logging file.
    log_path = os.path.join(exp_group_dir, 'test_metasgd.log')
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
    test_freq = 500
    
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
    
    # Iterating over folds.
    folds = [0, 1, 2, 3, 4]
    for fold in folds:
        
        print('Testing fold %d/%d...' % ((fold + 1), len(folds)))
        
        # Saving model and optimizer.
        model_ckpt_path = os.path.join(exp_ckpt_dir, 'model.pth')
        optim_ckpt_path = os.path.join(exp_ckpt_dir, 'optim.pth')
        
        if 'unet' in network_name:
            
            # Create model.
            model = UNet(1, unet_base, 2, final=True)
            if is_parallel: model = nn.DataParallel(model)
            model = l2l.algorithms.MetaSGD(model, lr=fast_lr, first_order=False)
            model.to(device)
            if is_parallel:
                model.module.load_state_dict(torch.load(model_ckpt_path).state_dict())
            else:
                model.load_state_dict(torch.load(model_ckpt_path).state_dict())
        
        elif 'efficientlab' in network_name:
            
            # Create model.
            model = EfficientLab(1, 2, final=True)
            if is_parallel: model = nn.DataParallel(model)
            model = l2l.algorithms.MetaSGD(model, lr=fast_lr, first_order=False)
            model.to(device)
            if is_parallel:
                model.module.load_state_dict(torch.load(model_ckpt_path).state_dict())
            else:
                model.load_state_dict(torch.load(model_ckpt_path).state_dict())
        
        elif 'deeplabv3' in network_name:
            
            # Create model.
            model = DeepLabV3(1, 2, final=True)
            if is_parallel: model = nn.DataParallel(model)
            model = l2l.algorithms.MetaSGD(model, lr=fast_lr, first_order=False)
            model.to(device)
            if is_parallel:
                model.module.load_state_dict(torch.load(model_ckpt_path).state_dict())
            else:
                model.load_state_dict(torch.load(model_ckpt_path).state_dict())
        
        elif 'resnet12' in network_name:
            
            # Create model.
            model = FCNResNet12(1, 2, final=True)
            if is_parallel: model = nn.DataParallel(model)
            model = l2l.algorithms.MetaSGD(model, lr=fast_lr, first_order=False)
            model.to(device)
            if is_parallel:
                model.module.load_state_dict(torch.load(model_ckpt_path).state_dict())
            else:
                model.load_state_dict(torch.load(model_ckpt_path).state_dict())
        
        # Setting optimizer and LR scheduler.
        opt = optim.Adam(model.parameters(), meta_lr)
        opt.load_state_dict(torch.load(optim_ckpt_path).state_dict())
        
        # Setting target datasets.
        exp_pnts, exp_grid, exp_scrb, exp_cntr, exp_skel, exp_poly, exp_dens = sup_qry_experiments(root,
                                                                                                   fold,
                                                                                                   trg_dataset,
                                                                                                   trg_task,
                                                                                                   crop_to,
                                                                                                   normalization,
                                                                                                   shots_list)
        
        # Testing the models.
        meta_test(model, adaptation_steps, loss_name, device, exp_group_dir, exp_pnts, 'points')
        meta_test(model, adaptation_steps, loss_name, device, exp_group_dir, exp_grid, 'grid')
        meta_test(model, adaptation_steps, loss_name, device, exp_group_dir, exp_scrb, 'scribbles')
        meta_test(model, adaptation_steps, loss_name, device, exp_group_dir, exp_cntr, 'contours')
        meta_test(model, adaptation_steps, loss_name, device, exp_group_dir, exp_skel, 'skeleton')
        meta_test(model, adaptation_steps, loss_name, device, exp_group_dir, exp_poly, 'polygons')
        meta_test(model, adaptation_steps, loss_name, device, exp_group_dir, exp_dens, 'dense')

if __name__ == '__main__':
    main()