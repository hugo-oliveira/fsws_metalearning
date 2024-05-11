import os
import sys
import time
import torch
import logging
import numpy as np
import nibabel as nib

from sparsify_3d import *

from torch.utils import data
from scipy import ndimage as ndi

from skimage import io
from skimage import draw
from skimage import util
from skimage import filters
from skimage import measure
from skimage import exposure
from skimage import transform
from skimage import morphology
from skimage import segmentation

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

# Class implementing the QueryDataset3D with sparse and dense labels.
class QueryDataset3D(data.Dataset):
    
    ################################################################
    # Initializing dataset.. #######################################
    ################################################################
    def __init__(self, root, fold, dataset, task, axis, resize_to=(128,128,128), normalization='z-score', zoom=0.5, debug=False, verbose=False):
        
        # Initializing variables.
        self.root = root
        self.fold = fold
        self.dataset = dataset
        self.task = task
        self.axis = axis
        self.resize_to = resize_to
        self.normalization = normalization
        self.zoom = zoom
        self.debug = debug
        self.verbose = verbose
        
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
        qry_data_list = sorted([l.strip('\n') for l in open(os.path.join(self.root, self.dataset, 'folds', 'tst_f%d.txt' % (self.fold))).readlines()])
        
        # Lists of query samples.
        qry_list = []
        
        # Populating lists with image and label paths for qry set.
        for it in qry_data_list:
            item = (os.path.join(img_dir, it),
                    os.path.join(msk_dir, it))
            qry_list.append(item)
        
        # Returning lists of samples for the dataset.
        return qry_list
    
    def get_task(self, path):
        
        lines = [l.strip('\n') for l in open(path).readlines() if l != '\n' and l != ' \n' and not l.startswith('#')]
        
        task = None
        
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
    def read_sample(self, img_path, msk_path):
        
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
        
        img, sparse_msk, msk, bundle = sparse_slices_3d(img, msk, sparsity_param={'num_slices': 1, 'axis': 2}, torch_random=True, seed=12345, debug=self.debug)
        
        # Adding channel dimension.
        img = np.expand_dims(img, axis=0)
        
        # Converting to torch tensors.
        img = torch.from_numpy(np.copy(img)).type(torch.FloatTensor)
        msk = torch.from_numpy(np.copy(msk)).type(torch.LongTensor)
        #sparse_msk = torch.from_numpy(np.copy(sparse_msk)).type(torch.LongTensor)
        
        return img, msk
    
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
        
        # Obtaining sample paths.
        img_path, msk_path = self.imgs[index]
        
        # Reading samples.
        qry_img, qry_msk = self.read_sample(img_path, msk_path)
        
        return qry_img, qry_msk, img_path.split('/')[-1]

    def __len__(self):

        return len(self.imgs)
