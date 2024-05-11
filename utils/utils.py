import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init

from sklearn import metrics

from skimage import io

###################################################
# Auxiliary functions. ############################
###################################################


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


def distance_logits(sup, qry):

#     print('sup', sup.shape) # [              train-way, n_out_channels]
#     print('qry', qry.shape) # [train-query * train-way, n_out_channels]
    
    size_sup = sup.shape[0]
    size_qry = qry.shape[0]
    
    # Computing logits by euclidean distance.
    sup_exp = sup.unsqueeze(0).expand(size_qry, size_sup, -1)
    qry_exp = qry.unsqueeze(1).expand(size_qry, size_sup, -1)
    
#     print('sup_exp', sup_exp.shape) # [train-query * train-way, train-way, n_out_channels]
#     print('qry_exp', qry_exp.shape) # [train-query * train-way, train-way, n_out_channels]
    
    logits = -((qry_exp - sup_exp)**2).sum(dim=2)
    
#     print('logits', logits.shape) # [train-query * train-way, train-way]
    
    return logits


def cosine_logits(trg, src_neg, src_pos):
    
    alpha_cosine = 20.0
    
    logits_neg = alpha_cosine * F.cosine_similarity(trg, src_neg)
    logits_pos = alpha_cosine * F.cosine_similarity(trg, src_pos)
    
    logits = torch.stack([logits_neg, logits_pos], dim=1)
    
    return logits


def accuracy(lab, prd):
    
    # Obtaining class from prediction.
    prd = prd.argmax(1)
    
    # Tensor to ndarray.
    lab_np = lab.view(-1).detach().cpu().numpy()
    prd_np = prd.view(-1).detach().cpu().numpy()
    
    # Computing metric and returning.
    metric_val = metrics.jaccard_score(lab_np, prd_np)
    
    return metric_val


def mode(ndarray, axis=0):
    # Check inputs
    ndarray = np.asarray(ndarray)
    ndim = ndarray.ndim
    if ndarray.size == 1:
        return (ndarray[0], 1)
    elif ndarray.size == 0:
        raise Exception('Cannot compute mode on empty array')
    try:
        axis = range(ndarray.ndim)[axis]
    except:
        raise Exception('Axis "{}" incompatible with the {}-dimension array'.format(axis, ndim))

    # If array is 1-D and numpy version is > 1.9 np.unique will suffice
    if all([ndim == 1,
            int(np.__version__.split('.')[0]) >= 1,
            int(np.__version__.split('.')[1]) >= 9]):
        modals, counts = np.unique(ndarray, return_counts=True)
        index = np.argmax(counts)
        return modals[index], counts[index]

    # Sort array
    sort = np.sort(ndarray, axis=axis)
    # Create array to transpose along the axis and get padding shape
    transpose = np.roll(np.arange(ndim)[::-1], axis)
    shape = list(sort.shape)
    shape[axis] = 1
    # Create a boolean array along strides of unique values
    strides = np.concatenate([np.zeros(shape=shape, dtype='bool'),
                                 np.diff(sort, axis=axis) == 0,
                                 np.zeros(shape=shape, dtype='bool')],
                                axis=axis).transpose(transpose).ravel()
    # Count the stride lengths
    counts = np.cumsum(strides)
    counts[~strides] = np.concatenate([[0], np.diff(counts[~strides])])
    counts[strides] = 0
    # Get shape of padded counts and slice to return to the original shape
    shape = np.array(sort.shape)
    shape[axis] += 1
    shape = shape[transpose]
    slices = [slice(None)] * ndim
    slices[axis] = slice(1, None)
    # Reshape and compute final counts
    counts = counts.reshape(shape).transpose(transpose)[slices] + 1

    # Find maximum counts and return modals/counts
    slices = [slice(None, i) for i in sort.shape]
    del slices[axis]
    index = np.ogrid[slices]
    index.insert(axis, np.argmax(counts, axis=axis))
    return sort[index], counts[index]


def collate_meta(batch):
    
    sup_img = []
    sup_msk = []
    sup_dns = []
    qry_img = []
    qry_msk = []
    sparsity_list = []
    sup_names = []
    qry_names = []
    
    for t in range(len(batch)):
        
        sup_img.append(batch[t][0])
        sup_msk.append(batch[t][1])
        sup_dns.append(batch[t][2])
        qry_img.append(batch[t][3])
        qry_msk.append(batch[t][4])
        sparsity_list.append(batch[t][5])
        sup_names.append(batch[t][6])
        qry_names.append(batch[t][7])
    
    return sup_img, sup_msk, sup_dns, qry_img, qry_msk, sparsity_list, sup_names, qry_names


def output_sup_results(exp_dir_sup, img, msk, dns, sample_name):
    
    # Iterating over batch.
    for b in range(img.size(0)):
        
        # Presetting paths.
        img_path = os.path.join(exp_dir_sup, sample_name[b].replace('.png', '_img.png'))
        msk_path = os.path.join(exp_dir_sup, sample_name[b].replace('.png', '_msk.png'))
        dns_path = os.path.join(exp_dir_sup, sample_name[b].replace('.png', '_dns.png'))
        
        # Post-processing images and masks.
        img_out = (((img[b] - img[b].min()) / (img[b].max() - img[b].min())) * 255).byte().detach().squeeze().cpu().numpy()
        msk_out = (msk[b] + 1).detach().squeeze().cpu().numpy().astype(np.uint8) * 127
        dns_out = (dns[b] + 1).detach().squeeze().cpu().numpy().astype(np.uint8) * 127
        
        # Saving images.
        io.imsave(img_path, img_out)
        io.imsave(msk_path, msk_out)
        io.imsave(dns_path, dns_out)


def output_qry_results(exp_dir_qry, img, msk, prd, sample_name):
    
    # Presetting paths.
    img_path = os.path.join(exp_dir_qry, sample_name.replace('.png', '_img.png'))
    msk_path = os.path.join(exp_dir_qry, sample_name.replace('.png', '_msk.png'))
    prd_path = os.path.join(exp_dir_qry, sample_name.replace('.png', '_prd.png'))
    
    # Post-processing images and masks.
    img_out = (((img - img.min()) / (img.max() - img.min())) * 255).byte().detach().squeeze().cpu().numpy()
    msk_out = (msk + 1).detach().squeeze().cpu().numpy().astype(np.uint8) * 127
    prd_out = (prd.max(dim=0)[1] + 1).detach().squeeze().cpu().numpy().astype(np.uint8) * 127
    
    # Saving images.
    io.imsave(img_path, img_out)
    io.imsave(msk_path, msk_out)
    io.imsave(prd_path, prd_out)



