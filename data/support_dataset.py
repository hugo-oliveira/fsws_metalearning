import os
import torch
import numpy as np

from sparsify import *

from torch.utils import data

from skimage import io
from skimage import exposure
from skimage import transform

'''
Fold structure for the 2d datasets:

root                              # root folder containing all datasets
|- jsrt                           # the primary folder of a dataset have its name (see MetaDataset.dataset_list, and change it to match the datasets in your root folder)
|  |- imgs                        # folder with all images
|  |- groundtruths                # folder with the masks for the images divide by task (e.g. task 1, contains the masks of the binary labels for task 1)
|  |  |- task 1
|  |  |- task 2
|  |  ...
|  |  |- task n
|  |- valid_labels.txt            # text file with the name of the tasks (one per line)
|  |- folds                       # folder with text files, where each file contains a list of the images either in the training or test set for fold k for the n-th task
|  |  |- trn_task1_f1.txt
|  |  |- tst_task1_f1.txt
|  |  |- trn_task1_f2.txt
|  |  |- tst_task1_f2.txt
|  |  ...
|  |  |- trn_taskn_fk.txt
|  |  |- tst_taskn_fk.txt
|- montgomery                     # other named datasets, that have the same subfolder organization as above
|- shenzhen
|- openist
...
|- panoramic
'''

# Class implementing the SupportDataset with sparse and dense labels.
class SupportDataset(data.Dataset):
    
    ################################################################
    # Initializing dataset. ########################################
    ################################################################
    def __init__(self, root, fold, dataset, task, sparsity_mode, sparsity_param, resize_to=(128,128), normalization='z-score', num_shots=2, debug=False, verbose=False):
        
        # Initializing variables.
        self.root = root
        self.fold = fold
        self.dataset = dataset
        self.task = task
        self.sparsity_mode = sparsity_mode
        self.sparsity_param = sparsity_param
        self.resize_to = resize_to
        self.normalization = normalization
        self.num_shots = num_shots
        self.debug = debug
        self.verbose = verbose
        
        # Presetting datasets and sparsity modes.
        self.sparsity_mode_list = ['points', 'grid', 'scribbles', 'contours', 'skeleton', 'polygons']
        assert self.sparsity_mode in self.sparsity_mode_list or self.sparsity_mode == 'dense', 'Sparsity mode "%s" not implemented. Exiting...' % (self.sparsity_mode)
        
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
        msk_dir = os.path.join(self.root, self.dataset, 'ground_truths', self.task)
        
        # Reading paths from file.
        sup_data_list = sorted([l.strip('\n') for l in open(os.path.join(self.root, self.dataset, 'folds', 'trn_%s_f%d.txt' % (self.task, self.fold))).readlines()])
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
        img = io.imread(img_path)
        msk = io.imread(msk_path)
        
        # Casting images to the appropriate dtypes.
        img = img.astype(np.float32)
        msk = msk.astype(np.int64)
        
        # Remove channel dimension.
        if len(img.shape) > 2:
            img = img[:,:,0]
        
        # Binarizing mask.
        msk[msk > 0] = 1
        
        # Trimming edges from images (when needed).
        img, msk = self.trim(img, msk)
        
        # Prior normalization.
        img = ((img - img.min()) / (img.max() - img.min()))
        
        # Slicing the volume.
        img, msk = self.resizing(img, msk)
        
        # Normalization.
        assert self.normalization in ['minmax', 'z-score', 'fixed']
        if self.normalization == 'minmax':
            img = ((img - img.min()) / (img.max() - img.min())) * 2 - 1.0
        elif self.normalization == 'z-score':
            img = (img - img.mean()) / img.std()
        elif self.normalization == 'fixed':
            img = (img - 0.4270778599819702) / 0.26350769943779223
        
        # Selecting annotation mode for the batch.
        bundle = None
        if self.sparsity_mode == 'points':
            sparse_msk, _ = sparse_points(msk, sparsity_param=self.sparsity_param, torch_random=True, seed=seed, debug=self.debug)
        elif self.sparsity_mode == 'grid':
            sparse_msk, _ = sparse_grid(msk, sparsity_param=self.sparsity_param, torch_random=True, seed=seed, debug=self.debug)
        elif self.sparsity_mode == 'scribbles':
            sparse_msk, _ = sparse_scribbles(msk, sparsity_param=self.sparsity_param, torch_random=True, seed=seed, debug=self.debug)
        elif self.sparsity_mode == 'contours':
            sparse_msk, _ = sparse_contours(msk, sparsity_param=self.sparsity_param, torch_random=True, seed=seed, debug=self.debug)
        elif self.sparsity_mode == 'skeleton':
            sparse_msk, _ = sparse_skeleton(msk, sparsity_param=self.sparsity_param, torch_random=True, seed=seed, debug=self.debug)
        elif self.sparsity_mode == 'polygons':
            sparse_msk, _ = sparse_polygon(msk, sparsity_param=self.sparsity_param, torch_random=True, seed=seed, debug=self.debug)
        elif self.sparsity_mode == 'dense':
            sparse_msk = np.copy(msk)
        
        # Adding channel dimension.
        img = np.expand_dims(img, axis=0)
        
        # Converting to torch tensors.
        img = torch.from_numpy(np.copy(img))
        msk = torch.from_numpy(np.copy(msk)).type(torch.LongTensor)
        sparse_msk = torch.from_numpy(np.copy(sparse_msk)).type(torch.LongTensor)
        
        return img, msk, sparse_msk
    
    ############################################################################################################################
    # Trim function adapted from: https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy #
    ############################################################################################################################
    def trim(self, img, msk):
        
        tolerance = 0.01 * float(img.max())
        
        # Mask of non-black pixels (assuming image has a single channel).
        bin = img > tolerance
        
        # Coordinates of non-black pixels.
        coords = np.argwhere(bin)
        
        # Bounding box of non-black pixels.
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top
        
        # Get the contents of the bounding box.
        if x1 - x0 < 16 or y1 - y0 < 16:
            
            return img, msk
            
        else:
            
            img_crop = img[x0:x1, y0:y1]
            msk_crop = msk[x0:x1, y0:y1]
            
            return img_crop, msk_crop
    
    ################################################################
    # Batch load function. #########################################
    ################################################################
    def __getitem__(self, index):
        
        # Allocating images, masks and sparse masks for support set.
        sup_img = torch.zeros((self.num_shots, 1, *self.resize_to), dtype=torch.float32)
        sup_msk = torch.zeros((self.num_shots, *self.resize_to), dtype=torch.long)
        sup_dns = torch.zeros((self.num_shots, *self.resize_to), dtype=torch.long)
        
        if self.verbose:
            print('Mode', self.sparsity_mode)
            print('Params', self.sparsity_param)
        
        # Sample names.
        sup_names = []
        
        # Iterating over support samples.
        for sup_i in range(self.num_shots):
            
            sample = self.imgs[sup_i]
            
            # Obtaining paths for images and masks.
            img_path, msk_path = sample
            
            sup_names.append(img_path.split('/')[-1])
            
            # Reading samples.
            img, msk, sparse_msk = self.read_sample(img_path, msk_path, seed=index)
            
            # Filling tensors.
            sup_img[sup_i, 0] = img
            sup_msk[sup_i] = sparse_msk
            sup_dns[sup_i] = msk
        
        return sup_img, sup_msk, sup_dns, sup_names
    
    def __len__(self):

        return 1 # Only 1 batch containing all samples is returned.
