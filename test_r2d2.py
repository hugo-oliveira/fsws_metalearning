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

from torch.nn import functional

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

from models.classification_heads import *

def fast_tune(sup_img,
              sup_msk,
              qry_loader,
              model,
              head,
              loss_name,
              device):
    
    # Time counters.
    adapt_time = 0.0
    test_time = 0.0
    
    # Setting model and head to evaluation mode (no tuning required on R2D2).
    model.eval()
    head.eval()
    with torch.no_grad():
        
        # Computing loss weights [1.0, 1.0].
        weights = loss_weights(sup_msk, device)
        
        # Tic.
        tic = time.time()
        
        # Linearizing support mask.
        lin_sup_msk = sup_msk.view(-1) # [B, H, W] -> [B * H * W]
        
        # Computing embeddings.
        sup_emb = model(sup_img)

        # Permuting embedding dimensions.
        sup_emb = sup_emb.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]

        # Linearizing batch and spacial dimensions of support and query.
        sup_emb = torch.reshape(sup_emb, (-1, sup_emb.shape[3])) # [B, H, W, C] -> [B * H * W, C]

        # Acquiring negative and positive samples from support.
        neg_sup = sup_emb[lin_sup_msk == 0]
        pos_sup = sup_emb[lin_sup_msk == 1]
        
        max_sup_size = 250
        
        # Undersampling overrepresented class on support set.
        if neg_sup.size(0) > pos_sup.size(0):
            neg_perm = torch.randperm(neg_sup.size(0))[:pos_sup.size(0)]
            neg_sup = neg_sup[neg_perm]
            pos_perm = torch.randperm(pos_sup.size(0))
            pos_sup = pos_sup[pos_perm]
        else:
            pos_perm = torch.randperm(pos_sup.size(0))[:neg_sup.size(0)]
            pos_sup = pos_sup[pos_perm]
            neg_perm = torch.randperm(neg_sup.size(0))
            neg_sup = neg_sup[neg_perm]
        
        # Undersampling tensors for CUDA memory.
        neg_sup = neg_sup[:min(neg_sup.size(0), max_sup_size)]
        pos_sup = pos_sup[:min(pos_sup.size(0), max_sup_size)]
        
        # Concating negative and positive prototypes on support.
        sup_emb = torch.cat([neg_sup, pos_sup], 0).unsqueeze(0)

        # Computing one-hot-encoded linear masks for support.
        lin_sup_msk = torch.zeros((sup_emb.shape[1]), dtype=torch.long).unsqueeze(0).to(device)
        lin_sup_msk[:, (lin_sup_msk.shape[1] // 2):] = 1
        
        # Toc.
        toc = time.time()
        adapt_time = (toc - tic)
        
        # Evaluate the adapted model.
        qry_err_list = []
        qry_met_list = []
        
        # Iterating over query batches.
        for batch in qry_loader:

            qry_img, qry_msk, qry_names = batch

            qry_img = qry_img.to(device)
            qry_msk = qry_msk.to(device)

            tic = time.time() # Tic.

            # Linearizing query mask.
            lin_qry_msk = qry_msk.view(-1) # [B, H, W] -> [B * H * W]

            # Computing embeddings for query sample.
            qry_emb = model(qry_img)

            qry_emb = qry_emb.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
            qry_emb = torch.reshape(qry_emb, (-1, qry_emb.shape[3])) # [B, H, W, C] -> [B * H * W, C]

            # Concating negative and positive prototypes on support.
            qry_emb = qry_emb.unsqueeze(0)
            
            logits_qry = head(qry_emb, sup_emb, lin_sup_msk, 2, sup_emb.size(1) // 2, 
                              l2_regularizer_lambda=10.0 if 'resnet50' in network_name or 'resnet18' in network_name else 1.0)
            
            toc = time.time() # Toc.
            test_time += (toc - tic)

            # Computing supervised loss on query predictions.
            logits_qry = logits_qry.squeeze(0)

            qry_err = loss_fn(logits_qry, lin_qry_msk, weights, loss_name, device)

            qry_met = accuracy(lin_qry_msk, logits_qry)

            if torch.any(torch.isnan(qry_err)):

                qry_err_list.append(0.0)
                qry_met_list.append(0.0)

                qry_prd = torch.zeros((2, qry_msk.shape[-2], qry_msk.shape[-1]), dtype=torch.float32)
                qry_prd[0, :, :] = 1.0

            else:

                qry_err_list.append(qry_err.item())
                qry_met_list.append(qry_met)

                qry_prd = logits_qry.view(qry_msk.shape[-2], qry_msk.shape[-1], 2).permute(2, 0, 1)

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
    
    # Reverting model and head to training mode.
    model.train()
    head.train()
    
    return qry_err_list, qry_met_list

def meta_test(model, head, loss_name, device, exp_group_dir, exp, sparsity_mode):
    
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
        
        # Recovering support dataset.
        sup_img, sup_msk, sup_dns, sup_names = sup['sup_dataset'][0]
        
        # Tuning and saving outputs.
        err_list, met_list = fast_tune(sup_img.to(device),
                                       sup_msk.to(device),
                                       qry_loader,
                                       copy.deepcopy(model),
                                       copy.deepcopy(head),
                                       loss_name,
                                       device)

def main(n_shots=5,             # Number of shots of each task.
         meta_lr=0.1,           # LR of the outer loop.
         inner_iters=4,         # Number of iterations in inner loop.
         outer_iters=6000,      # Number of iterations in outer loop.
         eps_smooth=0.1,        # Smoothing factor of one-hot-encoding on query labels.
         seed=42                # Random generator seed.
        ):
    
    # Recovering target dataset and task.
    assert len(sys.argv) >= 5, 'Not enough input parameters. Exiting...'
    
    global network_name
    trg_dataset = sys.argv[1]  # Target dataset for testing.
    trg_task = sys.argv[2]     # Target task for testing.
    network_name = sys.argv[3] # Segmentation network architecture.
    loss_name = sys.argv[4]    # Loss function.
    
    if 'resnet50' in network_name or 'resnet18' in network_name:
        
        meta_lr = 0.05
        eps_smooth = 0.01
    
    # Setting experiment name
    exp_group_name = 'r2d2_%s_%s_%s_%s' % (network_name, loss_name, trg_dataset, trg_task)
    
    # Setting result directory.
    result_dir = './experiments/'
    exp_group_dir = os.path.join(result_dir, exp_group_name)
    exp_ckpt_dir = os.path.join(result_dir, exp_group_name, 'ckpt')
    
    # Setting logging file.
    log_path = os.path.join(exp_group_dir, 'test_r2d2.log')
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
    
    shots_list = [1, 5, 10]
    # shots_list = [1, 5, 10, 20]
    
    # Iterating over folds.
    folds = [0, 1, 2, 3, 4]
    for fold in folds:
        
        print('Testing fold %d/%d...' % ((fold + 1), len(folds)))
        
        # Loading model and optimizer.
        model_ckpt_path = os.path.join(exp_ckpt_dir, 'model.pth')
        head_ckpt_path = os.path.join(exp_ckpt_dir, 'head.pth')
        optim_ckpt_path = os.path.join(exp_ckpt_dir, 'optim.pth')
        
        if 'unet' in network_name:
            
            # Create core model.
            model = UNet(1, unet_base, 2, final=False)
            model.to(device)
            model.load_state_dict(torch.load(model_ckpt_path).state_dict())
            
            # Create classification head.
            head = ClassificationHead(base_learner='R2D2')
            
            head.to(device)
            head.load_state_dict(torch.load(head_ckpt_path).state_dict())
        
        elif 'efficientlab' in network_name:
            
            # Create core model.
            model = EfficientLab(1, 2, final=False)
            model.to(device)
            model.load_state_dict(torch.load(model_ckpt_path).state_dict())
            
            # Create classification head.
            head = ClassificationHead(base_learner='R2D2')
            
            head.to(device)
            head.load_state_dict(torch.load(head_ckpt_path).state_dict())
        
        elif 'deeplabv3' in network_name:
            
            # Create core model.
            model = DeepLabV3(1, 2, final=False)
            model.to(device)
            model.load_state_dict(torch.load(model_ckpt_path).state_dict())
            
            # Create classification head.
            head = ClassificationHead(base_learner='R2D2')
            
            head.to(device)
            head.load_state_dict(torch.load(head_ckpt_path).state_dict())
        
        elif 'resnet12' in network_name:
            
            # Create core model.
            model = FCNResNet12(1, 2, final=False)
            model.to(device)
            model.load_state_dict(torch.load(model_ckpt_path).state_dict())
            
            # Create classification head.
            head = ClassificationHead(base_learner='R2D2')
            
            head.to(device)
            head.load_state_dict(torch.load(head_ckpt_path).state_dict())
        
        elif 'resnet18' in network_name:
            
            # Create core model.
            model = FCNResNet18(1, 2, final=False)
            model.to(device)
            model.load_state_dict(torch.load(model_ckpt_path).state_dict())
            
            # Create classification head.
            head = ClassificationHead(base_learner='R2D2')
            
            head.to(device)
            head.load_state_dict(torch.load(head_ckpt_path).state_dict())
        
        elif 'resnet50' in network_name:
            
            # Create core model.
            model = FCNResNet50(1, 2, final=False)
            model.to(device)
            model.load_state_dict(torch.load(model_ckpt_path).state_dict())
            
            # Create classification head.
            head = ClassificationHead(base_learner='R2D2')
            
            head.to(device)
            head.load_state_dict(torch.load(head_ckpt_path).state_dict())
        
        # Setting optimizer.
        opt = torch.optim.SGD([{'params': model.parameters()}, 
                               {'params': head.parameters()}], lr=meta_lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
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
        meta_test(model, head, loss_name, device, exp_group_dir, exp_pnts, 'points')
        meta_test(model, head, loss_name, device, exp_group_dir, exp_grid, 'grid')
        meta_test(model, head, loss_name, device, exp_group_dir, exp_scrb, 'scribbles')
        meta_test(model, head, loss_name, device, exp_group_dir, exp_cntr, 'contours')
        meta_test(model, head, loss_name, device, exp_group_dir, exp_skel, 'skeleton')
        meta_test(model, head, loss_name, device, exp_group_dir, exp_poly, 'polygons')
        meta_test(model, head, loss_name, device, exp_group_dir, exp_dens, 'dense')

if __name__ == '__main__':
    main()