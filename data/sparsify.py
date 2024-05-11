import torch
import numpy as np

from skimage import draw
from skimage import measure
from skimage import morphology

################################################################
# Sparse annotations through image transformations. ############
################################################################
def sparse_points(msk, sparsity_param, torch_random=True, seed=42, debug=False):

    # If no positive pixel at volume, return full mask.
    if not np.any(msk):
        return msk, {}
    
    # Fixing seed for random numbers.
    if torch_random:
        torch.manual_seed(seed)
    else:
        np.random.seed(seed)

    # Linearizing mask.
    msk_ravel = msk.ravel()

    # Slicing array for only containing negative class pixels.
    neg_msk = msk_ravel[msk_ravel == 0]
    neg_msk[:] = -1

    # Slicing array for only containing positive class pixels.
    pos_msk = msk_ravel[msk_ravel > 0]
    pos_msk[:] = -1

    # Negative mask.
    perm_neg = None
    if torch_random:
        perm_neg = torch.randperm(neg_msk.shape[0]).tolist() # Random perm of negative voxels.
    else:
        perm_neg = np.random.permutation(neg_msk.shape[0]) # Preset perm of negative voxels (fixed by seed).
    neg_msk[perm_neg[:min(sparsity_param['num_points'], len(perm_neg))]] = 0

    # Positive mask.
    if torch_random:
        perm_pos = torch.randperm(pos_msk.shape[0]).tolist() # Random perm of positive voxels.
    else:
        perm_pos = np.random.permutation(pos_msk.shape[0]) # Preset perm of positive voxels (fixed by seed).
    pos_msk[perm_pos[:min(sparsity_param['num_points'], len(perm_pos))]] = 1

    # Merging negative and positive sparse masks.
    new_msk = np.zeros(msk_ravel.shape[0], dtype=np.int8)
    new_msk[:] = -1
    new_msk[msk_ravel == 0] = neg_msk
    new_msk[msk_ravel  > 0] = pos_msk

    # Reshaping linearized sparse mask to the original 2 dimensions.
    new_msk = new_msk.reshape(msk.shape)

    # Dilating points.
    if sparsity_param['radius'] > 0:

        selem_thick = morphology.disk(sparsity_param['radius'])

        new_msk[morphology.binary_dilation(new_msk == 0, selem_thick) & (msk == 0)] = 0
        new_msk[morphology.binary_dilation(new_msk == 1, selem_thick) & (msk == 1)] = 1

    # Fixing potential annotation errors.
    new_msk[(new_msk == 1) & (msk == 0)] = -1
    new_msk[(new_msk == 0) & (msk == 1)] = -1

    # Debugging bundle.
    bundle = {}
    if debug:
        bundle = sparsity_param

    return new_msk, bundle

def sparse_grid(msk, sparsity_param, torch_random=True, seed=42, debug=False):
    
    # If no positive pixel at volume, return full mask.
    if not np.any(msk):
        return msk, {}
    
    # Fixing seed for random numbers.
    if torch_random:
        torch.manual_seed(seed)
    else:
        np.random.seed(seed)
    
    # Predetermined sparsity (y and x point spacing).
    spacing = (sparsity_param['space_y'],
               sparsity_param['space_x'])
    
    if torch_random:
        starting = (torch.randint(0, spacing[0], (1,)).item(),
                    torch.randint(0, spacing[1], (1,)).item()) # Random starting offset.
    else:
        starting = (np.random.randint(spacing[0]),
                    np.random.randint(spacing[1])) # Preset starting offset (fixed by seed).
    
    # Copying mask and starting it with -1 for inserting sparsity.
    new_msk = np.zeros_like(msk)
    new_msk[:, :] = -1
    new_msk[starting[0]::spacing[0],
            starting[1]::spacing[1]] = msk[starting[0]::spacing[0],
                                           starting[1]::spacing[1]]
    
    # Dilating points.
    if sparsity_param['radius'] > 0:
        
        selem_thick = morphology.disk(sparsity_param['radius'])
        
        new_msk[morphology.binary_dilation(new_msk == 0, selem_thick) & (msk == 0)] = 0
        new_msk[morphology.binary_dilation(new_msk == 1, selem_thick) & (msk == 1)] = 1
    
    # Fixing potential annotation errors.
    new_msk[(new_msk == 1) & (msk == 0)] = -1
    new_msk[(new_msk == 0) & (msk == 1)] = -1
    
    # Debugging bundle.
    bundle = {}
    if debug:
        bundle = sparsity_param
    
    return new_msk, bundle

def sparse_scribbles(msk, sparsity_param, torch_random=True, seed=42, debug=False):

    # If no positive pixel at volume, return full mask.
    if not np.any(msk):
        return msk, {}
    
    # Fixing seed for random numbers.
    if torch_random:
        torch.manual_seed(seed)
    else:
        np.random.seed(seed)
    
    # Sparsity proportion of contour.
    prop = sparsity_param['prop']
    
    # Dist radius for erosions and dilations from the original mask.
    radius_dist = sparsity_param['dist']
    
    # Disk radius for annotation thickness.
    radius_thick = sparsity_param['thick']
    
    # Dilating and eroding original mask and obtaining contours.
    selem_dist = morphology.disk(radius_dist)
    
    msk_neg = morphology.binary_dilation(msk > 0, selem_dist)
    msk_pos = morphology.binary_erosion(msk > 0, selem_dist)
    
    # If the organ is too small, the erosion might erase all positive pixels.
    # In this case use the contours of the original mask and let the label error correction deal with this later.
    if not np.any(msk_pos):
        msk_pos = msk > 0
    
    pos_contr = measure.find_contours(msk_pos) #, 0.0)
    neg_contr = measure.find_contours(msk_neg) #, 0.0)
    
    # Instantiating masks for the boundaries.
    msk_neg_bound = np.zeros_like(msk_neg)
    msk_pos_bound = np.zeros_like(msk_pos)
    
    # Filling boundary masks.
    for i, obj in enumerate(pos_contr):
        rand_rot = None
        if torch_random:
            rand_rot = torch.randint(low=1, high=len(obj), size=(1,)).item() # Random rotation of contour.
        else:
            rand_rot = np.random.randint(low=1, high=len(obj)) # Predefined rotation of contour (fixed by seed).
        for j, contour in enumerate(np.roll(obj, rand_rot, axis=0)):
            if j < max(1, round(len(obj) * prop)):
                msk_pos_bound[int(contour[0]), int(contour[1])] = 1
    
    for i, obj in enumerate(neg_contr):
        rand_rot = None
        if torch_random:
            rand_rot = torch.randint(low=1, high=len(obj), size=(1,)).item() # Random rotation of contour.
        else:
            rand_rot = np.random.randint(low=1, high=len(obj)) # Predefined rotation of contour (fixed by seed).
        for j, contour in enumerate(np.roll(obj, rand_rot, axis=0)):
            if j < max(1, round(len(obj) * prop)):
                msk_neg_bound[int(contour[0]), int(contour[1])] = 1
    
    # Performing dilation on the boundary masks for adding thickness.
    if radius_thick > 0:
        
        selem_thick = morphology.disk(radius_thick)
        
        msk_neg_bound = morphology.binary_dilation(msk_neg_bound, footprint=selem_thick)
        msk_pos_bound = morphology.binary_dilation(msk_pos_bound, footprint=selem_thick)
    
    # Fixing inconsistencies in borders.
    msk_neg_bound = msk_neg_bound & (msk == 0)
    msk_pos_bound = msk_pos_bound & (msk == 1)
    
    # Grouping positive, negative and unlabeled pixels.
    new_msk = np.zeros_like(msk, dtype=np.int8)
    new_msk[:] = -1
    new_msk[msk_neg_bound] = 0
    new_msk[msk_pos_bound] = 1
    
    # Fixing potential annotation errors.
    new_msk[(new_msk == 1) & (msk == 0)] = -1
    new_msk[(new_msk == 0) & (msk == 1)] = -1
    
    # Debugging bundle.
    bundle = {}
    if debug:
        bundle = sparsity_param
    
    return new_msk, bundle

def sparse_contours(msk, sparsity_param, torch_random=True, seed=42, debug=False):

    # If no positive pixel at volume, return full mask.
    if not np.any(msk):
        return msk, {}
    
    # Fixing seed for random numbers.
    if torch_random:
        torch.manual_seed(seed)
    else:
        np.random.seed(seed)
    
    # Sparsity proportion of contour.
    prop = sparsity_param['prop']
    
    # Disk radius for annotation thickness.
    radius_thick = sparsity_param['thick']
    
    # Finding contours of shape.
    contr = measure.find_contours(msk) #, 0.0)
    
    # Grouping positive, negative and unlabeled pixels.
    new_msk = np.zeros_like(msk, dtype=np.int8)
    new_msk[:] = -1
    
    # Filling boundary masks.
    for i, obj in enumerate(contr):
        rand_rot = None
        if torch_random:
            rand_rot = torch.randint(low=1, high=len(obj), size=(1,)).item() # Random rotation of contour.
        else:
            rand_rot = np.random.randint(low=1, high=len(obj)) # Predefined rotation of contour (fixed by seed).
        for j, contour in enumerate(np.roll(obj, rand_rot, axis=0)):
            if j < max(1, round(len(obj) * prop)):
                new_msk[int(contour[0]), int(contour[1])] = 1
    
    # Dilating border between classes.
    selem_thick = morphology.disk(radius_thick)
    contr_msk = morphology.binary_dilation(new_msk != -1, footprint=selem_thick)
    new_msk[contr_msk] = msk[contr_msk]
    
    # Fixing potential annotation errors.
    new_msk[(new_msk == 1) & (msk == 0)] = -1
    new_msk[(new_msk == 0) & (msk == 1)] = -1
    
    # Debugging bundle.
    bundle = {}
    if debug:
        bundle = sparsity_param
    
    return new_msk, bundle

def sparse_skeleton(msk, sparsity_param, torch_random=True, seed=42, debug=False):

    # If no positive pixel at volume, return full mask.
    if not np.any(msk):
        return msk, {}
    
    # Dist radius for erosions and dilations from the original mask.
    radius_dist = sparsity_param['dist']
    
    # Disk radius for annotation thickness.
    radius_thick = sparsity_param['thick']
    
    # Dilating and skeletonizing original mask and obtaining skeleton.
    selem_dist = morphology.disk(radius_dist)
    
    msk_neg = morphology.binary_dilation(msk.copy(order='C') > 0, selem_dist)
    msk_pos = morphology.skeletonize(msk.copy(order='C') > 0)
    
    pos_contr = measure.find_contours(msk_pos) #, 0.0)
    neg_contr = measure.find_contours(msk_neg) #, 0.0)
    
    # Instantiating masks for the boundaries.
    msk_neg_bound = np.zeros_like(msk_neg)
    msk_pos_bound = np.zeros_like(msk_pos)
    
    # Filling boundary masks.
    for i, obj in enumerate(pos_contr):
        rand_rot = None
        if torch_random:
            rand_rot = torch.randint(low=1, high=len(obj), size=(1,)).item() # Random rotation of contour.
        else:
            rand_rot = np.random.randint(low=1, high=len(obj)) # Predefined rotation of contour (fixed by seed).
        for j, contour in enumerate(np.roll(obj, rand_rot, axis=0)):
            msk_pos_bound[int(contour[0]), int(contour[1])] = 1
    
    for i, obj in enumerate(neg_contr):
        rand_rot = None
        if torch_random:
            rand_rot = torch.randint(low=1, high=len(obj), size=(1,)).item() # Random rotation of contour.
        else:
            rand_rot = np.random.randint(low=1, high=len(obj)) # Predefined rotation of contour (fixed by seed).
        for j, contour in enumerate(np.roll(obj, rand_rot, axis=0)):
            msk_neg_bound[int(contour[0]), int(contour[1])] = 1
    
    # Performing dilation on the boundary masks for adding thickness.
    if radius_thick > 0:
        
        selem_thick = morphology.disk(radius_thick)
        
        msk_neg_bound = morphology.binary_dilation(msk_neg_bound, footprint=selem_thick)
        msk_pos_bound = morphology.binary_dilation(msk_pos_bound, footprint=selem_thick)
    
    # Fixing inconsistencies in borders.
    msk_neg_bound = msk_neg_bound & (msk == 0)
    msk_pos_bound = msk_pos_bound & (msk == 1)
    
    # Grouping positive, negative and unlabeled pixels.
    new_msk = np.zeros_like(msk, dtype=np.int8)
    new_msk[:] = -1
    new_msk[msk_neg_bound] = 0
    new_msk[msk_pos_bound] = 1
    
    # Fixing potential annotation errors.
    new_msk[(new_msk == 1) & (msk == 0)] = -1
    new_msk[(new_msk == 0) & (msk == 1)] = -1
    
    # Debugging bundle.
    bundle = {}
    if debug:
        bundle = sparsity_param
    
    return new_msk, bundle

def sparse_polygon(msk, sparsity_param, torch_random=True, seed=42, debug=False):

    # If no positive pixel at volume, return full mask.
    if not np.any(msk):
        return msk, {}
    
    # Dist radius for erosions and dilations from the original outside mask.
    radius_dist = sparsity_param['dist']
    
    # Tolerances for polygon approximation.
    neg_tol = sparsity_param['tol_neg']
    pos_tol = sparsity_param['tol_pos']
    
    # Computing dilated convex hull for negative class.
    selem_dist = morphology.disk(radius_dist)
    
    msk_neg = morphology.convex_hull_image(msk > 0)
    msk_neg = morphology.binary_dilation(msk_neg, footprint=selem_dist)
    
    # Computing initial positive mask.
    msk_pos = msk > 0
    
    # Computing polygon for negative and positive classes.
    neg_contr = measure.find_contours(msk_neg) #, 0.0)
    pos_contr = measure.find_contours(msk_pos) #, 0.0)
    
    # Iterating over individual negative objects.
    msk_neg = np.zeros_like(msk_neg, dtype=bool)
    # msk_neg = np.zeros_like(msk_neg, dtype=np.bool)
    for cntr in neg_contr:
        
        # Computing polygon for negative class object(s).
        poly_neg = measure.approximate_polygon(cntr, tolerance=neg_tol)
        
        # Contour to mask.
        poly_neg_msk = draw.polygon2mask(msk_neg.shape, poly_neg)
        msk_neg[poly_neg_msk] = True
    
    # Inverting negative mask.
    msk_neg = ~msk_neg
    
    # Iterating over suitable tolerances for positive class.
    n_iters_tol = 5
    for tol_i in range(n_iters_tol):
        
        msk_pos = np.zeros_like(msk_pos, dtype=bool)
        
        # Iterating over individual positive objects.
        for cntr in pos_contr:
            
            # Computing polygon for positive class object(s).
            poly_pos = measure.approximate_polygon(cntr, tolerance=pos_tol)
            
            # Contour to mask.
            poly_pos_msk = draw.polygon2mask(msk_pos.shape, poly_pos)
            msk_pos[poly_pos_msk] = True
        
        if np.any(msk_pos):
            break
        else:
            pos_tol = pos_tol / 2.0
    
    if not np.any(msk_pos):
        msk_pos = msk > 0
    
    # Fixing inconsistencies in borders.
    msk_neg = msk_neg & (msk == 0)
    msk_pos = msk_pos & (msk == 1)
    
    # Grouping positive, negative and unlabeled pixels.
    new_msk = np.zeros_like(msk, dtype=np.int8)
    new_msk[:] = -1
    new_msk[msk_neg] = 0
    new_msk[msk_pos] = 1
    
    # Fixing potential annotation errors.
    new_msk[(new_msk == 1) & (msk == 0)] = -1
    new_msk[(new_msk == 0) & (msk == 1)] = -1
    
    # Debugging bundle.
    bundle = {}
    if debug:
        bundle = sparsity_param
    
    return new_msk, bundle