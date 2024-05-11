import torch
import numpy as np

from sparsify import *

def choose_slices(msk, num_slices, axis, torch_random=True, seed=42, debug=False):
    
    # Computing ROI for the positive class.
    all_props = measure.regionprops(msk)
    
    if len(all_props) > 0:
        
        props_3d = all_props[0]
        roi_3d = props_3d.bbox
        if axis == 0:
            mn_idx = roi_3d[0]
            mx_idx = roi_3d[3]
        elif axis == 1:
            mn_idx = roi_3d[1]
            mx_idx = roi_3d[4]
        elif axis == 2:
            mn_idx = roi_3d[2]
            mx_idx = roi_3d[5]
        
    else:
        
        mn_idx = 0
        mx_idx = msk.shape[axis]
    
    # Selecting equally espaced slice indices.
    slices = []
    slc_div = num_slices + 1
    slc_off = (mx_idx - mn_idx) / slc_div
    for s in range(1, slc_div):
        
        ind = mn_idx + int(round(s * slc_off))
        slices.append(ind)
    
    return slices

def sparse_points_3d(img, msk, sparsity_param, torch_random=True, seed=42, debug=False):
    
    # Number of weakly annotated slices.
    num_slices = sparsity_param['num_slices']
    
    # Axis to extract the slices.
    axis = sparsity_param['axis']
    
    # Fixing seed for random numbers.
    if torch_random:
        torch.manual_seed(seed)
    else:
        np.random.seed(seed)
    
    # Choosing slices for weak supervision.
    slices = choose_slices(msk, num_slices, axis, torch_random=torch_random, seed=seed, debug=debug)
    
    # Creating image, sparse/dense masks and parameter bundle lists.
    img_list = []
    msk_list = []
    dns_list = []
    bundle_list = []
    
    # Iterating over volume slices.
    for s in slices:
        
        if axis == 0:
            slc_img = img[s, :, :]
            slc_dns = msk[s, :, :]
        elif axis == 1:
            slc_img = img[:, s, :]
            slc_dns = msk[:, s, :]
        elif axis == 2:
            slc_img = img[:, :, s]
            slc_dns = msk[:, :, s]
        
        # Obtaining sparse version of 2D mask.
        slc_msk, bundle = sparse_points(slc_dns, sparsity_param, torch_random=torch_random, seed=seed, debug=debug)
        
        # Appending lists for slice image, slice mask and sparsity bundle.
        img_list.append(slc_img)
        msk_list.append(slc_msk)
        dns_list.append(slc_dns)
        bundle_list.append(bundle)
    
    if num_slices == 1:
        
        return img_list[0], msk_list[0], dns_list[0], bundle_list[0]
    else:
        
        return img_list, msk_list, dns_list, bundle_list

def sparse_grid_3d(img, msk, sparsity_param, torch_random=True, seed=42, debug=False):
    
    # Number of weakly annotated slices.
    num_slices = sparsity_param['num_slices']
    
    # Axis to extract the slices.
    axis = sparsity_param['axis']
    
    # Fixing seed for random numbers.
    if torch_random:
        torch.manual_seed(seed)
    else:
        np.random.seed(seed)
    
    # Choosing slices for weak supervision.
    slices = choose_slices(msk, num_slices, axis, torch_random=torch_random, seed=seed, debug=debug)
    
    # Creating image, sparse/dense masks and parameter bundle lists.
    img_list = []
    msk_list = []
    dns_list = []
    bundle_list = []
    
    # Iterating over volume slices.
    for s in slices:
        
        if axis == 0:
            slc_img = img[s, :, :]
            slc_dns = msk[s, :, :]
        elif axis == 1:
            slc_img = img[:, s, :]
            slc_dns = msk[:, s, :]
        elif axis == 2:
            slc_img = img[:, :, s]
            slc_dns = msk[:, :, s]
        
        # Obtaining sparse version of 2D mask.
        slc_msk, bundle = sparse_grid(slc_dns, sparsity_param, torch_random=torch_random, seed=seed, debug=debug)
        
        # Appending lists for slice image, slice mask and sparsity bundle.
        img_list.append(slc_img)
        msk_list.append(slc_msk)
        dns_list.append(slc_dns)
        bundle_list.append(bundle)
    
    if num_slices == 1:
        
        return img_list[0], msk_list[0], dns_list[0], bundle_list[0]
    else:
        
        return img_list, msk_list, dns_list, bundle_list

def sparse_scribbles_3d(img, msk, sparsity_param, torch_random=True, seed=42, debug=False):
    
    # Number of weakly annotated slices.
    num_slices = sparsity_param['num_slices']
    
    # Axis to extract the slices.
    axis = sparsity_param['axis']
    
    # Fixing seed for random numbers.
    if torch_random:
        torch.manual_seed(seed)
    else:
        np.random.seed(seed)
    
    # Choosing slices for weak supervision.
    slices = choose_slices(msk, num_slices, axis, torch_random=torch_random, seed=seed, debug=debug)
    
    # Creating image, sparse/dense masks and parameter bundle lists.
    img_list = []
    msk_list = []
    dns_list = []
    bundle_list = []
    
    # Iterating over volume slices.
    for s in slices:
        
        if axis == 0:
            slc_img = img[s, :, :]
            slc_dns = msk[s, :, :]
        elif axis == 1:
            slc_img = img[:, s, :]
            slc_dns = msk[:, s, :]
        elif axis == 2:
            slc_img = img[:, :, s]
            slc_dns = msk[:, :, s]
        
        # Obtaining sparse version of 2D mask.
        slc_msk, bundle = sparse_scribbles(slc_dns, sparsity_param, torch_random=torch_random, seed=seed, debug=debug)
        
        # Appending lists for slice image, slice mask and sparsity bundle.
        img_list.append(slc_img)
        msk_list.append(slc_msk)
        dns_list.append(slc_dns)
        bundle_list.append(bundle)
    
    if num_slices == 1:
        
        return img_list[0], msk_list[0], dns_list[0], bundle_list[0]
    else:
        
        return img_list, msk_list, dns_list, bundle_list

def sparse_contours_3d(img, msk, sparsity_param, torch_random=True, seed=42, debug=False):
    
    # Number of weakly annotated slices.
    num_slices = sparsity_param['num_slices']
    
    # Axis to extract the slices.
    axis = sparsity_param['axis']
    
    # Fixing seed for random numbers.
    if torch_random:
        torch.manual_seed(seed)
    else:
        np.random.seed(seed)
    
    # Choosing slices for weak supervision.
    slices = choose_slices(msk, num_slices, axis, torch_random=torch_random, seed=seed, debug=debug)
    
    # Creating image, sparse/dense masks and parameter bundle lists.
    img_list = []
    msk_list = []
    dns_list = []
    bundle_list = []
    
    # Iterating over volume slices.
    for s in slices:
        
        if axis == 0:
            slc_img = img[s, :, :]
            slc_dns = msk[s, :, :]
        elif axis == 1:
            slc_img = img[:, s, :]
            slc_dns = msk[:, s, :]
        elif axis == 2:
            slc_img = img[:, :, s]
            slc_dns = msk[:, :, s]
        
        # Obtaining sparse version of 2D mask.
        slc_msk, bundle = sparse_contours(slc_dns, sparsity_param, torch_random=torch_random, seed=seed, debug=debug)
        
        # Appending lists for slice image, slice mask and sparsity bundle.
        img_list.append(slc_img)
        msk_list.append(slc_msk)
        dns_list.append(slc_dns)
        bundle_list.append(bundle)
    
    if num_slices == 1:
        
        return img_list[0], msk_list[0], dns_list[0], bundle_list[0]
    else:
        
        return img_list, msk_list, dns_list, bundle_list

def sparse_skeleton_3d(img, msk, sparsity_param, torch_random=True, seed=42, debug=False):
    
    # Number of weakly annotated slices.
    num_slices = sparsity_param['num_slices']
    
    # Axis to extract the slices.
    axis = sparsity_param['axis']
    
    # Fixing seed for random numbers.
    if torch_random:
        torch.manual_seed(seed)
    else:
        np.random.seed(seed)
    
    # Choosing slices for weak supervision.
    slices = choose_slices(msk, num_slices, axis, torch_random=torch_random, seed=seed, debug=debug)
    
    # Creating image, sparse/dense masks and parameter bundle lists.
    img_list = []
    msk_list = []
    dns_list = []
    bundle_list = []
    
    # Iterating over volume slices.
    for s in slices:
        
        if axis == 0:
            slc_img = img[s, :, :]
            slc_dns = msk[s, :, :]
        elif axis == 1:
            slc_img = img[:, s, :]
            slc_dns = msk[:, s, :]
        elif axis == 2:
            slc_img = img[:, :, s]
            slc_dns = msk[:, :, s]
        
        # Obtaining sparse version of 2D mask.
        slc_msk, bundle = sparse_skeleton(slc_dns, sparsity_param, torch_random=torch_random, seed=seed, debug=debug)
        
        # Appending lists for slice image, slice mask and sparsity bundle.
        img_list.append(slc_img)
        msk_list.append(slc_msk)
        dns_list.append(slc_dns)
        bundle_list.append(bundle)
    
    if num_slices == 1:
        
        return img_list[0], msk_list[0], dns_list[0], bundle_list[0]
    else:
        
        return img_list, msk_list, dns_list, bundle_list

def sparse_polygon_3d(img, msk, sparsity_param, torch_random=True, seed=42, debug=False):
    
    # Number of weakly annotated slices.
    num_slices = sparsity_param['num_slices']
    
    # Axis to extract the slices.
    axis = sparsity_param['axis']
    
    # Fixing seed for random numbers.
    if torch_random:
        torch.manual_seed(seed)
    else:
        np.random.seed(seed)
    
    # Choosing slices for weak supervision.
    slices = choose_slices(msk, num_slices, axis, torch_random=torch_random, seed=seed, debug=debug)
    
    # Creating image, sparse/dense masks and parameter bundle lists.
    img_list = []
    msk_list = []
    dns_list = []
    bundle_list = []
    
    # Iterating over volume slices.
    for s in slices:
        
        if axis == 0:
            slc_img = img[s, :, :]
            slc_dns = msk[s, :, :]
        elif axis == 1:
            slc_img = img[:, s, :]
            slc_dns = msk[:, s, :]
        elif axis == 2:
            slc_img = img[:, :, s]
            slc_dns = msk[:, :, s]
        
        # Obtaining sparse version of 2D mask.
        slc_msk, bundle = sparse_polygon(slc_dns, sparsity_param, torch_random=torch_random, seed=seed, debug=debug)
        
        # Appending lists for slice image, slice mask and sparsity bundle.
        img_list.append(slc_img)
        msk_list.append(slc_msk)
        dns_list.append(slc_dns)
        bundle_list.append(bundle)
    
    if num_slices == 1:
        
        return img_list[0], msk_list[0], dns_list[0], bundle_list[0]
    else:
        
        return img_list, msk_list, dns_list, bundle_list

def sparse_slices_3d(img, msk, sparsity_param, torch_random=True, seed=42, debug=False):
    
    # Number of weakly annotated slices.
    num_slices = sparsity_param['num_slices']
    
    # Axis to extract the slices.
    axis = sparsity_param['axis']
    
    # Fixing seed for random numbers.
    if torch_random:
        torch.manual_seed(seed)
    else:
        np.random.seed(seed)
    
    # Choosing slices for weak supervision.
    slices = choose_slices(msk, num_slices, axis, torch_random=torch_random, seed=seed, debug=debug)
    
    # Creating image, sparse/dense masks and parameter bundle lists.
    img_list = []
    msk_list = []
    dns_list = []
    bundle_list = []
    
    # Iterating over volume slices.
    for s in slices:
        
        if axis == 0:
            slc_img = img[s, :, :]
            slc_dns = msk[s, :, :]
        elif axis == 1:
            slc_img = img[:, s, :]
            slc_dns = msk[:, s, :]
        elif axis == 2:
            slc_img = img[:, :, s]
            slc_dns = msk[:, :, s]
        
        # Obtaining sparse version of 2D mask.
        slc_msk = np.copy(slc_dns)
        bundle = sparsity_param
        
        # Appending lists for slice image, slice mask and sparsity bundle.
        img_list.append(slc_img)
        msk_list.append(slc_msk)
        dns_list.append(slc_dns)
        bundle_list.append(bundle)
    
    if num_slices == 1:
        
        return img_list[0], msk_list[0], dns_list[0], bundle_list[0]
    else:
        
        return img_list, msk_list, dns_list, bundle_list