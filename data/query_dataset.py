import os
import sys
import time
import torch
import numpy as np

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

# Class implementing the QueryDataset with sparse and dense labels.
class QueryDataset(data.Dataset):
    
    ################################################################
    # Initializing dataset. ########################################
    ################################################################
    def __init__(self, root, fold, dataset, task, resize_to=(128,128), normalization='z-score', debug=False, verbose=False):
        
        # Initializing variables.
        self.root = root
        self.fold = fold
        self.dataset = dataset
        self.task = task
        self.resize_to = resize_to
        self.normalization = normalization
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
        msk_dir = os.path.join(self.root, self.dataset, 'ground_truths', self.task)
        
        # Reading paths from file.
        qry_data_list = [l.strip('\n') for l in open(os.path.join(self.root, self.dataset, 'folds', 'tst_%s_f%d.txt' % (self.task, self.fold))).readlines()]
        
        # Lists of query samples.
        qry_list = []
        
        # Populating lists with image and label paths for qry set.
        for it in qry_data_list:
            item = (os.path.join(img_dir, it),
                    os.path.join(msk_dir, it))
            qry_list.append(item)
        
        # Returning lists of samples for the dataset.
        return qry_list
    
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
        
        # Adding channel dimension.
        img = np.expand_dims(img, axis=0).astype(np.float32)
        
        # Converting to torch tensors.
        img = torch.from_numpy(np.copy(img))
        msk = torch.from_numpy(np.copy(msk)).type(torch.LongTensor)
        
        return img, msk
    
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
        
        # Obtaining sample paths.
        img_path, msk_path = self.imgs[index]
        
        # Reading samples.
        qry_img, qry_msk = self.read_sample(img_path, msk_path)
        
        return qry_img, qry_msk, img_path.split('/')[-1]
    
    def __len__(self):

        return len(self.imgs)
