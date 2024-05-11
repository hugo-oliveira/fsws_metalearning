import os
import torch
import logging
import numpy as np
import nibabel as nib

from sparsify_3d import *

from torch.utils import data
from skimage import exposure
from skimage import transform

'''
Fold structure for the 3d datasets:

root                              # root folder containing all datasets
|- dataset1                       # the primary folder of a dataset have its name (see MetaDataset3D.dataset_list, and change it to match the datasets in your root folder)
|  |- imgs                        # folder with all images
|  |- groundtruths                # folder with the binary masks for the images
|  |- valid_labels.txt            # text file with the name of the task (one line)
|  |- folds                       # folder with text files, where each file contains a list of the images either in the training or test set for fold k
|  |  |- trn_f1.txt
|  |  |- tst_f1.txt
|  |  |- trn_f2.txt
|  |  |- tst_f2.txt
|  |  ...
|  |  |- trn_fk.txt
|  |  |- tst_fk.txt
|- dataset2                       # other named datasets, that have the same subfolder organization as above
|- dataset3
...
|- datasetM
'''

# Class implementing the SupportDataset3D with sparse and dense labels.
class SupportDataset3D(data.Dataset):
    
    ################################################################
    # Initializing dataset.. #######################################
    ################################################################
    def __init__(self, root, fold, dataset, task, sparsity_mode, sparsity_param, resize_to=(128,128,128), normalization='z-score', zoom=0.5, num_shots=2, debug=False, verbose=False):
        
        # Initializing variables.
        self.root = root
        self.fold = fold
        self.dataset = dataset
        self.task = task
        self.tsk_dict = None #Will be set ahead on self.make_dataset
        self.sparsity_mode = sparsity_mode
        self.sparsity_param = sparsity_param
        self.resize_to = resize_to
        self.normalization = normalization
        self.zoom = zoom
        self.num_shots = num_shots
        self.debug = debug
        self.verbose = verbose
        
        # Presetting datasets and sparsity modes.
        self.sparsity_mode_list = ['points', 'grid', 'scribbles', 'contours', 'skeleton', 'polygons']

        assert self.sparsity_mode in self.sparsity_mode_list or self.sparsity_mode == 'dense' or self.sparsity_mode == 'slices', 'Sparsity mode "%s" not implemented. Exiting...' % (self.sparsity_mode)
        
        # Presetting dataset.
        self.imgs = self.make_dataset()
        
        # Check for consistency in list.
        assert len(self.imgs) > 0, 'Found 0 images for dataset "%s", task "%s". Exiting...' % (self.dataset, self.task)
    
    ################################################################
    # Reading sample and valid task lists from files. ##############
    ################################################################
    def make_dataset(self):
        
        # Presetting paths.
        img_dir = os.path.join(self.root, self.dataset, 'images')
        msk_dir = os.path.join(self.root, self.dataset, 'ground_truths')
        tsk_path = os.path.join(self.root, self.dataset, 'valid_labels.txt')
        
        # Reading valid labels from file.
        task_dict = self.get_task(tsk_path)
        # Check for consistency.
        assert task_dict is not None, 'Task %s not found in dataset %s. Exiting...' % (self.task,self.dataset)
        self.tsk_dict = task_dict
        
        # Reading paths from file.
        sup_data_list = sorted([l.strip('\n') for l in open(os.path.join(self.root, self.dataset, 'folds', 'trn_f%d.txt' % (self.fold))).readlines()])
        np.random.seed(42)
        perm = np.random.permutation(len(sup_data_list))[:min(len(sup_data_list), self.num_shots)]
        
        # Lists of support samples.
        sup_list = []
        
        # Populating lists with image and label paths for sup set.
        for p in perm:
            it = sup_data_list[p]
            item = (os.path.join(img_dir, it),
                    os.path.join(msk_dir, it))
            sup_list.append(item)
        
        # Returning lists of samples for the dataset.
        return sup_list
    
    def get_task(self, path):
        
        lines = [l.strip('\n') for l in open(path).readlines() if l != '\n' and l != ' \n' and not l.startswith('#')]
        
        task = {}
        
        for l in lines:
            
            tsk_name = l.split(': ')[0]
            if tsk_name != self.task:
                continue
            tsk_src = l.split(': ')[-1].split('->')[0]
            tsk_trg = int(l.split('->')[-1])
            
            if '|' in tsk_src:
                tsk_src = [int(s) for s in tsk_src.split('|')]
            else:
                tsk_src = [int(tsk_src)]
            task = {'name': tsk_name,
                    'src': tsk_src,
                    'trg': tsk_trg}
            break
        
        return task
    
    ################################################################
    # Resizing. ####################################################
    ################################################################
    def resizing(self, img, msk):
        
        # Resizing to size self.resize_to.
        img = transform.resize(img, self.resize_to, order=1, preserve_range=True, anti_aliasing=False)
        msk = transform.resize(msk, self.resize_to, order=0, preserve_range=True, anti_aliasing=False).astype(np.int64)
        
        # Equalizing histogram.
        img = exposure.equalize_adapthist(img)
        
        return img, msk
    
    ################################################################
    # Reading and preprocessing samples. ###########################
    ################################################################
    def read_sample(self, img_path, msk_path, seed):
        
        # Reading images.
        img = nib.load(img_path).get_fdata()
        msk = nib.load(msk_path).get_fdata()
        
        # Casting images to the appropriate dtypes.
        img = img.astype(np.float32)
        msk = msk.astype(np.int64)
        
        # Clipping extreme values.
        img = np.clip(img, a_min=np.quantile(img, 0.005), a_max=np.quantile(img, 0.95))
        
        # Selecting task indices.
        src_indices = self.tsk_dict['src']
        trg_index = self.tsk_dict['trg']
        new_msk = np.zeros_like(msk, dtype=msk.dtype)
        for i in src_indices:
            new_msk[msk == i] = trg_index
        
        msk = new_msk
        
        # Binarizing mask.
        msk[msk != 0] = 1
        
        # Fixed zoom into organ ROI.
        img, msk = self.apply_zoom(img, msk)
        
        # Slicing the volume.
        img = ((img - img.min()) / (img.max() - img.min())) * 2 - 1.0
        img, msk = self.resizing(img, msk)
        
        # Normalization.
        assert self.normalization in ['minmax', 'z-score']
        if self.normalization == 'minmax':
            img = ((img - img.min()) / (img.max() - img.min())) * 2 - 1.0
        elif self.normalization == 'z-score':
            img = (img - img.mean()) / img.std()
        
        # Selecting annotation mode for the batch.
        bundle = None
        if self.sparsity_mode == 'points':
            img, sparse_msk, msk, bundle = sparse_points_3d(img, msk, sparsity_param=self.sparsity_param, torch_random=True, seed=seed, debug=self.debug)
        elif self.sparsity_mode == 'grid':
            img, sparse_msk, msk, bundle = sparse_grid_3d(img, msk, sparsity_param=self.sparsity_param, torch_random=True, seed=seed, debug=self.debug)
        elif self.sparsity_mode == 'scribbles':
            img, sparse_msk, msk, bundle = sparse_scribbles_3d(img, msk, sparsity_param=self.sparsity_param, torch_random=True, seed=seed, debug=self.debug)
        elif self.sparsity_mode == 'contours':
            img, sparse_msk, msk, bundle = sparse_contours_3d(img, msk, sparsity_param=self.sparsity_param, torch_random=True, seed=seed, debug=self.debug)
        elif self.sparsity_mode == 'skeleton':
            img, sparse_msk, msk, bundle = sparse_skeleton_3d(img, msk, sparsity_param=self.sparsity_param, torch_random=True, seed=seed, debug=self.debug)
        elif self.sparsity_mode == 'polygons':
            img, sparse_msk, msk, bundle = sparse_polygon_3d(img, msk, sparsity_param=self.sparsity_param, torch_random=True, seed=seed, debug=self.debug)
        elif self.sparsity_mode == 'dense' or self.sparsity_mode == 'slices':
            img, sparse_msk, msk, bundle = sparse_slices_3d(img, msk, sparsity_param=self.sparsity_param, torch_random=True, seed=seed, debug=self.debug)
        
        # Adding channel dimension.
        img = np.expand_dims(img, axis=0)
        
        # Converting to torch tensors.
        img = torch.from_numpy(np.copy(img)).type(torch.FloatTensor)
        msk = torch.from_numpy(np.copy(msk)).type(torch.LongTensor)
        sparse_msk = torch.from_numpy(np.copy(sparse_msk)).type(torch.LongTensor)
        
        return img, sparse_msk, msk, bundle
    
    ################################################################
    # Applying zoom around segmentation ROI. #######################
    ################################################################
    def apply_zoom(self, img, msk):
        
        # Mask of ROI pixels.
        bin = msk > 0
        
        # Coordinates of non-black pixels.
        coords = np.argwhere(bin)
        
        # Bounding box of non-black pixels.
        x0, y0, z0 = coords.min(axis=0)
        x1, y1, z1 = coords.max(axis=0) + 1   # slices are exclusive at the top
        
        if self.verbose:
            print('Old ROI: (%d..%d, %d..%d, %d..%d)' % (x0,
                                                         x1,
                                                         y0,
                                                         y1,
                                                         z0,
                                                         z1))
            logging.info('Old ROI: (%d..%d, %d..%d, %d..%d)' % (x0,
                                                                x1,
                                                                y0,
                                                                y1,
                                                                z0,
                                                                z1))
        
        margins_3d = (int(round(x0 * (1.0 - self.zoom))),
                      int(round(y0 * (1.0 - self.zoom))),
                      int(round(z0 * (1.0 - self.zoom))),
                      int(round((msk.shape[0] - x1) * (1.0 - self.zoom))),
                      int(round((msk.shape[1] - y1) * (1.0 - self.zoom))),
                      int(round((msk.shape[2] - z1) * (1.0 - self.zoom))))
        
        new_roi_3d = ((x0 - margins_3d[0]),
                      (y0 - margins_3d[1]),
                      (z0 - margins_3d[2]),
                      (x1 + margins_3d[3]),
                      (y1 + margins_3d[4]),
                      (z1 + margins_3d[5]))
        
        if self.verbose:
            print('New ROI: (%d..%d, %d..%d, %d..%d)' % (new_roi_3d[0],
                                                         new_roi_3d[3],
                                                         new_roi_3d[1],
                                                         new_roi_3d[4],
                                                         new_roi_3d[2],
                                                         new_roi_3d[5]))
            logging.info('New ROI: (%d..%d, %d..%d, %d..%d)' % (new_roi_3d[0],
                                                                new_roi_3d[3],
                                                                new_roi_3d[1],
                                                                new_roi_3d[4],
                                                                new_roi_3d[2],
                                                                new_roi_3d[5]))
        
        new_img = img[new_roi_3d[0]:new_roi_3d[3],
                      new_roi_3d[1]:new_roi_3d[4],
                      new_roi_3d[2]:new_roi_3d[5]]
        new_msk = msk[new_roi_3d[0]:new_roi_3d[3],
                      new_roi_3d[1]:new_roi_3d[4],
                      new_roi_3d[2]:new_roi_3d[5]]
        
        return new_img, new_msk
        
    ################################################################
    # Batch load function. #########################################
    ################################################################
    def __getitem__(self, index):
        
        # Allocating images, masks and sparse masks for query and support sets.
        sup_img = torch.zeros((self.num_shots, 1, *self.resize_to[:2]), dtype=torch.float32)
        sup_msk = torch.zeros((self.num_shots, *self.resize_to[:2]), dtype=torch.long)
        sup_dns = torch.zeros((self.num_shots, *self.resize_to[:2]), dtype=torch.long)
        
        if self.verbose:
            print('Mode', self.sparsity_mode)
            print('Params', self.sparsity_param)
        
        # Volume names.
        sup_names = []
        
        # Iterating over support samples.
        for sup_i in range(self.num_shots):
            
            sample = self.imgs[sup_i]
            
            # Obtaining paths for images and masks.
            img_path, msk_path = sample
            
            sup_names.append(img_path.split('/')[-1])
            
            # Reading samples.
            img, sparse_msk, msk, bundle = self.read_sample(img_path, msk_path, seed=index)
            
            # Filling tensors.
            sup_img[sup_i, 0] = img
            sup_msk[sup_i] = sparse_msk
            sup_dns[sup_i] = msk
        
        return sup_img, sup_msk, sup_dns, sup_names

    def __len__(self):

        return 1 # Only 1 batch containing all samples is returned.
