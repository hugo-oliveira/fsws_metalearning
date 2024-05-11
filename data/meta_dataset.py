import os
import time
import torch
import logging
import numpy as np

from sparsify import *

from torch.utils import data
from scipy import ndimage as ndi

from skimage import io
from skimage import measure
from skimage import exposure
from skimage import transform

'''
Fold structure for the 2d datasets:

root                              # root folder containing all datasets
|- jsrt                           # the primary folder of a dataset have its name (see MetaDataset.dataset_list, and change it to match the datasets in your root folder)
|  |- imgs                        # folder with all images
|  |- groundtruths                # folder with the masks for the images divide by task subfolders (e.g. task 1, contains the masks of the binary labels for task 1)
|  |  |- task1                      # subfolder with task1 masks
|  |  |- task2
|  |  ...
|  |  |- task n
|  |- valid_labels.txt            # text file with the name of the tasks (one per line)
|  |- folds                       # folder with text files, where each file contains a list of the images either in the training or test set for fold k for the n-th task
|  |  |- trn_task1_f1.txt           # text file with the list of images in training in fold 1 for task 1
|  |  |- tst_task1_f1.txt           # text file with the list of images in test in fold 1 for task 1
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


# Class implementing the MetaDataset with sparse and dense labels.
class MetaDataset(data.Dataset):
    
    ################################################################
    # Initializing dataset. ########################################
    ################################################################
    def __init__(self, root, fold, trg_dataset, resize_to=(140,140), crop_to=(128,128), normalization='z-score', num_shots=2, only_dense=False, debug=False, verbose=False):
        
        # Initializing variables.
        self.root = root
        self.fold = fold
        self.crop_to = crop_to
        self.resize_to = resize_to
        self.normalization = normalization
        self.num_shots = num_shots
        self.only_dense = only_dense
        self.debug = debug
        self.verbose = verbose
        
        # Presetting datasets and sparsity modes.
        if self.only_dense:
            self.sparsity_mode_list = ['dense']
        else:
            self.sparsity_mode_list = ['points', 'grid', 'scribbles', 'contours', 'skeleton', 'polygons']

        self.dataset_list = [
            'jsrt',
            'montgomery',
            'shenzhen',
            'openist',
            'nih_labeled',
            'mias',
            'inbreast',
            'ivisionlab',
            'panoramic',
        ]
        self.dataset_list = [d for d in self.dataset_list if d != trg_dataset]
        
        # Creating lists of tasks for each dataset.
        self.tsks = []
        
        # Iterating over datasets.
        for i, d in enumerate(self.dataset_list):
            
            if self.verbose:
                print('Configuring dataset %d/%d: "%s"' % (i + 1, len(self.dataset_list), d))
                logging.info('Configuring dataset %d/%d: "%s"' % (i + 1, len(self.dataset_list), d))
            
            sup_list, qry_list, tasks = self.make_dataset(d)
            
            for task_data in zip(sup_list, qry_list, tasks):
                
                self.tsks.append({'sup': task_data[0],
                                  'qry': task_data[1],
                                  'index': i,
                                  'dataset': d, 
                                  'task': task_data[2]})
        
        # Check for consistency in list.
        assert len(self.tsks) > 0, 'Found 0 tasks at meta-dataset. Exiting...'
    
    ################################################################
    # Reading sample and valid task lists from files. ##############
    ################################################################
    def make_dataset(self, dataset):
        
        # Initiating empty list of samples.
        sup_list = []
        qry_list = []
        
        # Valid labels path.
        tsk_path = os.path.join(self.root, dataset, 'valid_labels.txt')
        
        # Reading valid labels from file.
        tasks = self.get_tasks(tsk_path)
        
        # Iterating over tasks.
        for t in tasks:
            
            # Presetting paths.
            img_dir = os.path.join(self.root, dataset, 'images')
            msk_dir = os.path.join(self.root, dataset, 'ground_truths', t)
            
            # Reading paths from file.
            sup_data_list = [l.strip('\n') for l in open(os.path.join(self.root, dataset, 'folds', 'trn_%s_f%d.txt' % (t, self.fold))).readlines()]
            qry_data_list = [l.strip('\n') for l in open(os.path.join(self.root, dataset, 'folds', 'tst_%s_f%d.txt' % (t, self.fold))).readlines()]
            
            # Lists of support and query samples.
            curr_sup = []
            curr_qry = []
            
            # Populating lists with image and label paths for qry and sup sets.
            for it in sup_data_list:
                item = (os.path.join(img_dir, it),
                        os.path.join(msk_dir, it))
                curr_sup.append(item)
            
            for it in qry_data_list:
                item = (os.path.join(img_dir, it),
                        os.path.join(msk_dir, it))
                curr_qry.append(item)
            
            sup_list.append(curr_sup)
            qry_list.append(curr_qry)
            
        # Returning lists of samples and tasks for the dataset.
        return sup_list, qry_list, tasks
    
    def get_tasks(self, path):
        
        tasks = [l.strip('\n') for l in open(path).readlines() if l != '\n' and l != ' \n' and l != '\n' and not l.startswith('#')]
        
        return tasks
    
    ################################################################
    # Slicing and Stitching. #######################################
    ################################################################
    def slicing_trn(self, img, msk, augmentation, torch_random=True):
        
        # Subsampling using stride.
        y_stride = (img.shape[0] // self.resize_to[0]) + 1
        x_stride = (img.shape[1] // self.resize_to[1]) + 1
        
        y_off = None
        x_off = None
        
        if torch_random:
            y_off = torch.randint(0, y_stride, (1,)).item() # Random offset.
            x_off = torch.randint(0, x_stride, (1,)).item() # Random offset.
        else:
            y_off = np.random.randint(y_stride) # Preset offset (fixed by seed).
            x_off = np.random.randint(x_stride) # Preset offset (fixed by seed).
        
        # Computing strided image and label.
        img = img[y_off::y_stride, x_off::x_stride]
        msk = msk[y_off::y_stride, x_off::x_stride]
        
        # Cropping to size self.resize_to.
        img = transform.resize(img, self.resize_to, order=1, preserve_range=True, anti_aliasing=False)
        msk = transform.resize(msk, self.resize_to, order=0, preserve_range=True, anti_aliasing=False).astype(np.int64)
        
        # Cropping to size self.crop_to.
        img = img[augmentation['offset_0']:(augmentation['offset_0'] + self.crop_to[0]),
                  augmentation['offset_1']:(augmentation['offset_1'] + self.crop_to[1])]
        msk = msk[augmentation['offset_0']:(augmentation['offset_0'] + self.crop_to[0]),
                  augmentation['offset_1']:(augmentation['offset_1'] + self.crop_to[1])]
        
        # Randomly flipping image on axis 1.
        if augmentation['flip_0']:
            img = np.flip(img, axis=0)
            msk = np.flip(msk, axis=0)
        if augmentation['flip_1']:
            img = np.flip(img, axis=1)
            msk = np.flip(msk, axis=1)
        
        # Random rotation of 90, 180 or 270 degrees.
        angle_k = augmentation['rot_90']
        if angle_k > 0:
            img = np.rot90(img, k=angle_k)
            msk = np.rot90(msk, k=angle_k)
        
        # Randomly rotating the volume for a few degrees across the axial plane.
        angle = augmentation['rot_angle']
        if angle != 0.0:
            img = ndi.rotate(img, angle, axes=(0,1), order=1, reshape=False)
            msk = ndi.rotate(msk, angle, axes=(0,1), order=0, reshape=False)
        
        # Equalizing histogram.
        if augmentation['equalize']:
            img = exposure.equalize_adapthist(img)
        
        return img, msk
    
    ################################################################
    # Reading and preprocessing samples. ###########################
    ################################################################
    def read_sample(self, img_path, msk_path, augmentation, sparsity_mode, sparsity_param, seed):
        
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
        
        # Random zoom into organ ROI.
        img, msk = self.apply_zoom(img, msk, augmentation)
        
        # Prior normalization.
        img = ((img - img.min()) / (img.max() - img.min()))
        
        # Slicing the volume.
        img, msk = self.slicing_trn(img, msk, augmentation, torch_random=True)
        
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
        if sparsity_mode == 'points':
            sparse_msk, bundle = sparse_points(msk, sparsity_param=sparsity_param, torch_random=True, seed=seed, debug=self.debug)
        elif sparsity_mode == 'grid':
            sparse_msk, bundle = sparse_grid(msk, sparsity_param=sparsity_param, torch_random=True, seed=seed, debug=self.debug)
        elif sparsity_mode == 'scribbles':
            sparse_msk, bundle = sparse_scribbles(msk, sparsity_param=sparsity_param, torch_random=True, seed=seed, debug=self.debug)
        elif sparsity_mode == 'contours':
            sparse_msk, bundle = sparse_contours(msk, sparsity_param=sparsity_param, torch_random=True, seed=seed, debug=self.debug)
        elif sparsity_mode == 'skeleton':
            sparse_msk, bundle = sparse_skeleton(msk, sparsity_param=sparsity_param, torch_random=True, seed=seed, debug=self.debug)
        elif sparsity_mode == 'polygons':
            sparse_msk, bundle = sparse_polygon(msk, sparsity_param=sparsity_param, torch_random=True, seed=seed, debug=self.debug)
        elif sparsity_mode == 'dense':
            sparse_msk = np.copy(msk)
        
        # Adding channel dimension.
        img = np.expand_dims(img, axis=0)
        
        # Converting to torch tensors.
        img = torch.from_numpy(np.copy(img))
        msk = torch.from_numpy(np.copy(msk)).type(torch.LongTensor)
        sparse_msk = torch.from_numpy(np.copy(sparse_msk)).type(torch.LongTensor)
        
        return img, msk, sparse_msk, bundle
    
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
        if x1 - x0 < 32 or y1 - y0 < 32:

            return img, msk
        
        else:
            
            img_crop = img[x0:x1, y0:y1]
            msk_crop = msk[x0:x1, y0:y1]

            return img_crop, msk_crop
    
    ################################################################
    # Applying zoom around segmentation ROI. #######################
    ################################################################
    def apply_zoom(self, img, msk, augmentation):
        
        all_props = measure.regionprops(msk)
        if len(all_props) == 0:
            return img, msk
        
        props_2d = all_props[0]
        roi_2d = props_2d.bbox
        
        if self.verbose:
            print('Old ROI: (%d..%d, %d..%d)' % (roi_2d[0],
                                                 roi_2d[2],
                                                 roi_2d[1],
                                                 roi_2d[3]))
            logging.info('Old ROI: (%d..%d, %d..%d)' % (roi_2d[0],
                                                        roi_2d[2],
                                                        roi_2d[1],
                                                        roi_2d[3]))
        
        margins_2d = (int(round(roi_2d[0] * (1.0 - augmentation['zoom']))),
                      int(round(roi_2d[1] * (1.0 - augmentation['zoom']))),
                      int(round((msk.shape[0] - roi_2d[2]) * (1.0 - augmentation['zoom']))),
                      int(round((msk.shape[1] - roi_2d[3]) * (1.0 - augmentation['zoom']))))
        
        new_roi_2d = ((roi_2d[0] - margins_2d[0]),
                      (roi_2d[1] - margins_2d[1]),
                      (roi_2d[2] + margins_2d[2]),
                      (roi_2d[3] + margins_2d[3]))
        
        if self.verbose:
            print('New ROI: (%d..%d, %d..%d)' % (new_roi_2d[0],
                                                 new_roi_2d[2],
                                                 new_roi_2d[1],
                                                 new_roi_2d[3]))
            logging.info('New ROI: (%d..%d, %d..%d)' % (new_roi_2d[0],
                                                        new_roi_2d[2],
                                                        new_roi_2d[1],
                                                        new_roi_2d[3]))
        
        new_img = img[new_roi_2d[0]:new_roi_2d[2],
                      new_roi_2d[1]:new_roi_2d[3]]
        new_msk = msk[new_roi_2d[0]:new_roi_2d[2],
                      new_roi_2d[1]:new_roi_2d[3]]
        
        return new_img, new_msk
    
    ################################################################
    # Precomputing data augmentation parameters for batch. #########
    ################################################################
    def random_augmentation(self, torch_random=True):
        
        augmentation = None
        if torch_random:
            augmentation = {'zoom': torch.rand(1).item() * 0.2,
                            'offset_0': torch.randint(0, self.resize_to[0] - self.crop_to[0], (1,)).item(),
                            'offset_1': torch.randint(0, self.resize_to[1] - self.crop_to[1], (1,)).item(),
                            'flip_0': torch.rand(1).item() > 0.5,
                            'flip_1': torch.rand(1).item() > 0.5,
                            'rot_90': torch.randint(0, 4, (1,)).item(),
                            'rot_angle': torch.randn(1).item() * 4 if torch.rand(1).item() > 0.3 else 0.0,
                            'equalize': True}
#                             'equalize': torch.rand(1).item() > 0.5}
        else:
            augmentation = {'zoom': np.random.random() * 0.2,
                            'offset_0': np.random.randint(self.resize_to[0] - self.crop_to[0]),
                            'offset_1': np.random.randint(self.resize_to[1] - self.crop_to[1]),
                            'flip_0': np.random.random() > 0.5,
                            'flip_1': np.random.random() > 0.5,
                            'rot_90': np.random.randint(4),
                            'rot_angle': np.random.randn() * 4 if np.random.random() > 0.3 else 0.0,
                            'equalize': np.random.random() > 0.5}
        
        return augmentation
        
    ################################################################
    # Batch load function. #########################################
    ################################################################
    def __getitem__(self, index):
        
        # Updating seed.
        time_ns = time.time_ns()
        seed = int(time_ns // (index + 1)) % 4294967296
        
        if self.verbose:
            print('time ns:', time_ns, ', index:', index, ', seed:', seed)
            logging.info('time ns: ' + str(time_ns) + ', index: ' + str(index) + ', seed:' + str(seed))
        
#         torch.manual_seed(seed)
        np.random.seed(seed)
        
        tsk_index = index
        
        # Allocating images, masks and sparse masks for query and support sets.
        sup_img = torch.zeros((self.num_shots, 1, *self.crop_to), dtype=torch.float32)
        qry_img = torch.zeros((self.num_shots, 1, *self.crop_to), dtype=torch.float32)
        sup_msk = torch.zeros((self.num_shots, *self.crop_to), dtype=torch.long)
        qry_msk = torch.zeros((self.num_shots, *self.crop_to), dtype=torch.long)
        sup_dns = torch.zeros((self.num_shots, *self.crop_to), dtype=torch.long)
        
        # Presetting variables from dict.
        tsk_dict = self.tsks[tsk_index]
        sup_len = len(tsk_dict['sup'])
        qry_len = len(tsk_dict['qry'])
        
        # Randomly selecting samples.
        perm_sup = np.random.permutation(sup_len)[:self.num_shots]
        perm_qry = np.random.permutation(qry_len)[:self.num_shots]
        
        # Precomputing data augmentation for the whole batch.
        augmentation = self.random_augmentation(torch_random=True)
        if self.verbose:
            print('Augmentation params:', augmentation)
            logging.info('Augmentation params: ' + str(augmentation))
        
        # Randomly selecing sparsity mode and params.
        sparsity_mode = np.random.randint(len(self.sparsity_mode_list))
        sparsity_mode = self.sparsity_mode_list[sparsity_mode]
        
        sparsity_param = None
        
        if sparsity_mode == 'points':
            
            # Randomly choosing number of labeled points.
            sparsity_param = {'num_points': np.random.randint(low=1, high=21),
                              'radius': np.random.randint(low=1, high=3)}
        
        elif sparsity_mode == 'grid':
            
            # Randomly choosing x and y point spacing.
            sparsity_param = {'space_y': np.random.randint(low=4, high=21),
                              'space_x': np.random.randint(low=4, high=21),
                              'radius': np.random.randint(low=1, high=2)}
        
        elif sparsity_mode == 'scribbles':
            
            # Randomly choosing scribble proportion, radius and thickness.
            sparsity_param = {'prop': np.random.random(),
                              'dist': np.random.randint(low=2, high=6),
                              'thick': np.random.randint(low=1, high=4)}
        
        elif sparsity_mode == 'contours':
            
            # Randomly choosing contour proportion, radius and thickness.
            sparsity_param = {'prop': np.random.random(),
                              'thick': np.random.randint(low=1, high=5)}
        
        elif sparsity_mode == 'skeleton':
            
            # Randomly choosing skeleton radius and thickness.
            sparsity_param = {'dist': np.random.randint(low=2, high=8),
                              'thick': np.random.randint(low=1, high=2)}
        
        elif sparsity_mode == 'polygons':
            
            # Randomly choosing polygon radius and tolerance.
            sparsity_param = {'dist': np.random.randint(low=2, high=6),
                              'tol_neg': (np.random.random() * 20.0) + 20.0,
                              'tol_pos': (np.random.random() * 10.0) + 5.0}
        
        elif sparsity_mode == 'dense':
            
            # No weak mask data augmentation.
            sparsity_param = {}
        
        if self.verbose:
            print('Mode', sparsity_mode)
            print('Params', sparsity_param)
            logging.info('Mode ' + str(sparsity_mode))
            logging.info('Params ' + str(sparsity_param))
        
        # Initiating support and query bundles.
        curr_bundle = {'task': tsk_dict['dataset'] + ' ' + tsk_dict['task'],
                       'mode': sparsity_mode,
                       'sup': []}
        
        # Support and query names.
        sup_names = []
        qry_names = []
        
        # Iterating over support samples.
        for sup_i, ps in enumerate(perm_sup):
            
            # Obtaining paths for images and masks.
            img_path, msk_path = tsk_dict['sup'][ps]
            
            sup_names.append(img_path.split('/')[-1])
            
            # Reading samples.
            img, msk, sparse_msk, bundle = self.read_sample(img_path, msk_path, augmentation, sparsity_mode, sparsity_param, seed)
            
            # Filling tensors.
            sup_img[sup_i, 0] = img
            sup_msk[sup_i] = sparse_msk
            sup_dns[sup_i] = msk
            
            # Populating bundle.
            curr_bundle['sup'].append(bundle)
            
        # Iterating over query samples.
        for qry_i, pq in enumerate(perm_qry):
            
            # Obtaining paths for images and masks.
            img_path, msk_path = tsk_dict['qry'][pq]
            
            qry_names.append(img_path.split('/')[-1])
            
            # Reading samples.
            img, msk, sparse_msk, bundle = self.read_sample(img_path, msk_path, augmentation, 'dense', None, seed)
            
            # Filling tensors.
            qry_img[qry_i, 0] = img
            qry_msk[qry_i] = msk
        
        return sup_img, sup_msk, sup_dns, qry_img, qry_msk, curr_bundle, sup_names, qry_names
    
    def __len__(self):

        return len(self.tsks)
