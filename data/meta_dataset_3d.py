import os
import time
import torch
import logging
import numpy as np
import nibabel as nib

from sparsify_3d import *

from torch.utils import data
from scipy import ndimage as ndi

from skimage import measure
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

# Class implementing the MetaDataset3D with sparse and dense labels.
class MetaDataset3D(data.Dataset):
    
    ################################################################
    # Initializing dataset.. #######################################
    ################################################################
    def __init__(self, root, fold, trg_dataset, resize_to=(140,140,140), crop_to=(128,128,128), normalization='z-score', num_shots=2, debug=False, verbose=False):
        
        # Initializing variables.
        self.root = root
        self.fold = fold
        self.crop_to = crop_to
        self.resize_to = resize_to
        self.normalization = normalization
        self.num_shots = num_shots
        self.debug = debug
        self.verbose = verbose
        
        # Presetting datasets and sparsity modes.
        self.sparsity_mode_list = ['points', 'grid', 'scribbles', 'contours', 'skeleton', 'polygons']
        self.dataset_list = [
            'covid19_ct_seg',
            'ct_org',
            'ctpel',  
            'decathlon_liver', 
            'decathlon_pancreas',
            'decathlon_spleen',
            'feta', 
            'kits21',
            'luna',
            'multiorgan_ct_btcv',
            'multiorgan_ct_pancreas',
            'qubiq_pancreas',
            'structseg_head',
            'structseg_thorax',
            'vessel12'
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
            msk_dir = os.path.join(self.root, dataset, 'ground_truths')
            
            # Reading paths from file.
            sup_data_list = [l.strip('\n') for l in open(os.path.join(self.root, dataset, 'folds', 'trn_f%d.txt' % (self.fold))).readlines()]
            qry_data_list = [l.strip('\n') for l in open(os.path.join(self.root, dataset, 'folds', 'tst_f%d.txt' % (self.fold))).readlines()]
            
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
        
        lines = [l.strip('\n') for l in open(path).readlines() if l != '\n' and l != ' \n' and not l.startswith('#')]
        
        tasks = []
        
        for l in lines:
            
            tsk_name = l.split(': ')[0]
            tsk_src = l.split(': ')[-1].split('->')[0]
            tsk_trg = int(l.split('->')[-1])
            
            if '|' in tsk_src:
                tsk_src = [int(s) for s in tsk_src.split('|')]
            else:
                tsk_src = [int(tsk_src)]
            
            if tsk_name != 'background':
                tasks.append({'name': tsk_name,
                              'src': tsk_src,
                              'trg': tsk_trg})
        
        return tasks
    
    ################################################################
    # Slicing and Stitching. #######################################
    ################################################################
    def slicing_trn(self, img, msk, augmentation, torch_random=True):
        
        # Subsampling using stride.
        y_stride = (img.shape[0] // self.resize_to[0]) + 1
        x_stride = (img.shape[1] // self.resize_to[1]) + 1
        z_stride = (img.shape[2] // self.resize_to[2]) + 1
        
        y_off = None
        x_off = None
        z_off = None
        
        if torch_random:
            y_off = torch.randint(0, y_stride, (1,)).item() # Random offset.
            x_off = torch.randint(0, x_stride, (1,)).item() # Random offset.
            z_off = torch.randint(0, z_stride, (1,)).item() # Random offset.
        else:
            y_off = np.random.randint(y_stride) # Preset offset (fixed by seed).
            x_off = np.random.randint(x_stride) # Preset offset (fixed by seed).
            z_off = np.random.randint(z_stride) # Preset offset (fixed by seed).
        
        # Computing strided image and label.
        img = img[y_off::y_stride, x_off::x_stride, z_off::z_stride]
        msk = msk[y_off::y_stride, x_off::x_stride, z_off::z_stride]
        
        # Cropping to size self.resize_to.
        img = transform.resize(img, self.resize_to, order=1, preserve_range=True, anti_aliasing=False)
        msk = transform.resize(msk, self.resize_to, order=0, preserve_range=True, anti_aliasing=False).astype(np.int64)
        
        # Cropping to size self.crop_to.
        img = img[augmentation['offset_0']:(augmentation['offset_0'] + self.crop_to[0]),
                  augmentation['offset_1']:(augmentation['offset_1'] + self.crop_to[1]),
                  augmentation['offset_2']:(augmentation['offset_2'] + self.crop_to[2])]
        msk = msk[augmentation['offset_0']:(augmentation['offset_0'] + self.crop_to[0]),
                  augmentation['offset_1']:(augmentation['offset_1'] + self.crop_to[1]),
                  augmentation['offset_2']:(augmentation['offset_2'] + self.crop_to[2])]
        
        # Randomly flipping image on axis 1.
        if augmentation['flip_0']:
            img = np.flip(img, axis=0)
            msk = np.flip(msk, axis=0)
        if augmentation['flip_1']:
            img = np.flip(img, axis=1)
            msk = np.flip(msk, axis=1)
        if augmentation['flip_2']:
            img = np.flip(img, axis=2)
            msk = np.flip(msk, axis=2)
        
        # Setting image to int (Lucas)
        img = img.astype(np.uint16)
        # Randomly rotating the volume for a few degrees across the axial plane.
        angle = augmentation['rot_angle']
        if angle != 0.0:
            img = ndi.rotate(img, angle, axes=augmentation['rot_axes'], order=1, reshape=False)
            msk = ndi.rotate(msk, angle, axes=augmentation['rot_axes'], order=0, reshape=False)
        
        # Equalizing histogram.
        if augmentation['equalize']:
            img = exposure.equalize_adapthist(img)
        
        return img, msk
    
    ################################################################
    # Reading and preprocessing samples. ###########################
    ################################################################
    def read_sample(self, img_path, msk_path, src_indices, trg_index, augmentation, sparsity_mode, sparsity_param, seed):
        
        # Reading images.
        img = nib.load(img_path).get_fdata()
        msk = nib.load(msk_path).get_fdata()
        
        # Casting images to the appropriate dtypes.
        img = img.astype(np.float32)
        msk = msk.astype(np.int64)
        
        # Clipping extreme values.
        img = np.clip(img, a_min=np.quantile(img, 0.005), a_max=np.quantile(img, 0.95))
        
        # Selecting task indices.
        new_msk = np.zeros_like(msk, dtype=msk.dtype)
        for i in src_indices:
            new_msk[msk == i] = trg_index
            
        msk = new_msk
        
        # Binarizing mask.
        msk[msk != 0] = 1
        
        # Random zoom into organ ROI.
        img, msk = self.apply_zoom(img, msk, augmentation)
        
        # Slicing the volume.
        img, msk = self.slicing_trn(img, msk, augmentation)
        
        # Normalization.
        assert self.normalization in ['minmax', 'z-score']
        if self.normalization == 'minmax':
            img = ((img - img.min()) / (img.max() - img.min())) * 2 - 1.0
        elif self.normalization == 'z-score':
            img = (img - img.mean()) / img.std()
        
        # Selecting annotation mode for the batch.
        bundle = None
        #print('sparsity_mode', sparsity_mode)
        #print('sparsity_param', sparsity_param)
        if sparsity_mode == 'points':
            img, sparse_msk, msk, bundle = sparse_points_3d(img, msk, sparsity_param=sparsity_param)
        elif sparsity_mode == 'grid':
            img, sparse_msk, msk, bundle = sparse_grid_3d(img, msk, sparsity_param=sparsity_param)
        elif sparsity_mode == 'scribbles':
            img, sparse_msk, msk, bundle = sparse_scribbles_3d(img, msk, sparsity_param=sparsity_param)
        elif sparsity_mode == 'contours':
            img, sparse_msk, msk, bundle = sparse_contours_3d(img, msk, sparsity_param=sparsity_param)
        elif sparsity_mode == 'skeleton':
            img, sparse_msk, msk, bundle = sparse_skeleton_3d(img, msk, sparsity_param=sparsity_param)
        elif sparsity_mode == 'polygons':
            img, sparse_msk, msk, bundle = sparse_polygon_3d(img, msk, sparsity_param=sparsity_param)
        elif sparsity_mode == 'dense' or sparsity_mode == 'slices':
            img, sparse_msk, msk, bundle = sparse_slices_3d(img, msk, sparsity_param=sparsity_param)
        
        # Adding channel dimension.
        img = np.expand_dims(img, axis=0)
        
        # Converting to torch tensors.
        img = torch.from_numpy(np.copy(img))
        msk = torch.from_numpy(np.copy(msk)).type(torch.LongTensor)
        sparse_msk = torch.from_numpy(np.copy(sparse_msk)).type(torch.LongTensor)
        
        return img, msk, sparse_msk, bundle
    
    ################################################################
    # Applying zoom around segmentation ROI. #######################
    ################################################################
    def apply_zoom(self, img, msk, augmentation):
        
        all_props = measure.regionprops(msk)
        if len(all_props) == 0:
            return img, msk
            
        props_3d = all_props[0]
        roi_3d = props_3d.bbox
        
        if self.verbose:
            print('Old ROI: (%d..%d, %d..%d, %d..%d)' % (roi_3d[0],
                                                         roi_3d[3],
                                                         roi_3d[1],
                                                         roi_3d[4],
                                                         roi_3d[2],
                                                         roi_3d[5]))
            logging.info('Old ROI: (%d..%d, %d..%d, %d..%d)' % (roi_3d[0],
                                                                roi_3d[3],
                                                                roi_3d[1],
                                                                roi_3d[4],
                                                                roi_3d[2],
                                                                roi_3d[5]))
        
        margins_3d = (int(round(roi_3d[0] * (1.0 - augmentation['zoom']))),
                      int(round(roi_3d[1] * (1.0 - augmentation['zoom']))),
                      int(round(roi_3d[2] * (1.0 - augmentation['zoom']))),
                      int(round((msk.shape[0] - roi_3d[3]) * (1.0 - augmentation['zoom']))),
                      int(round((msk.shape[1] - roi_3d[4]) * (1.0 - augmentation['zoom']))),
                      int(round((msk.shape[2] - roi_3d[5]) * (1.0 - augmentation['zoom']))))
        
        new_roi_3d = ((roi_3d[0] - margins_3d[0]),
                      (roi_3d[1] - margins_3d[1]),
                      (roi_3d[2] - margins_3d[2]),
                      (roi_3d[3] + margins_3d[3]),
                      (roi_3d[4] + margins_3d[4]),
                      (roi_3d[5] + margins_3d[5]))
        
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
    # Precomputing data augmentation parameters for batch. #########
    ################################################################
    def random_augmentation(self, torch_random=True):
        
        augmentation = None
        if torch_random:
            augmentation = {'zoom': torch.rand(1).item() * 0.2,
                            'offset_0': torch.randint(0, self.resize_to[0] - self.crop_to[0], (1,)).item(),
                            'offset_1': torch.randint(0, self.resize_to[1] - self.crop_to[1], (1,)).item(),
                            'offset_2': torch.randint(0, self.resize_to[2] - self.crop_to[2], (1,)).item(),
                            'flip_0': torch.rand(1).item() > 0.5,
                            'flip_1': torch.rand(1).item() > 0.5,
                            'flip_2': torch.rand(1).item() > 0.5,
                            'rot_axes': set(torch.randperm(3).tolist()[:2]),
                            'rot_angle': torch.randn(1).item() * 4 if torch.rand(1).item() > 0.3 else 0.0,
                            'equalize': True}
        else:
            augmentation = {'zoom': np.random.random() * 0.2,
                            'offset_0': np.random.randint(self.resize_to[0] - self.crop_to[0]),
                            'offset_1': np.random.randint(self.resize_to[1] - self.crop_to[1]),
                            'offset_2': np.random.randint(self.resize_to[2] - self.crop_to[2]),
                            'flip_0': np.random.random() > 0.5,
                            'flip_1': np.random.random() > 0.5,
                            'flip_2': np.random.random() > 0.5,
                            'rot_axes': set(np.random.permutation(3)[:2]),
                            'rot_angle': np.random.randn() * 4 if np.random.random() > 0.3 else 0.0,
                            'equalize': True}
        
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
        sup_img = torch.zeros((self.num_shots, 1, *self.crop_to[:2]), dtype=torch.float32)
        qry_img = torch.zeros((self.num_shots, 1, *self.crop_to[:2]), dtype=torch.float32)
        sup_msk = torch.zeros((self.num_shots, *self.crop_to[:2]), dtype=torch.long)
        qry_msk = torch.zeros((self.num_shots, *self.crop_to[:2]), dtype=torch.long)
        sup_dns = torch.zeros((self.num_shots, *self.crop_to[:2]), dtype=torch.long)
        
        # Presetting variables from dict.
        tsk_dict = self.tsks[tsk_index]
        sup_len = len(tsk_dict['sup'])
        qry_len = len(tsk_dict['qry'])
        src_indices = tsk_dict['task']['src']
        trg_index = tsk_dict['task']['trg']
            
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
            
            # Randomly choosing number of labeled points; number of slices and axis.
            sparsity_param = {'num_slices': 1, #np.random.randint(low=1, high=6),
                              'axis': np.random.randint(low=0, high=3),
                              'num_points': np.random.randint(low=1, high=21),
                              'radius': np.random.randint(low=1, high=3)}
        
        elif sparsity_mode == 'grid':
            
            # Randomly choosing x and y point spacing; number of slices and axis.
            sparsity_param = {'num_slices': 1, #np.random.randint(low=1, high=6),
                              'axis': np.random.randint(low=0, high=3),
                              'space_y': np.random.randint(low=4, high=21),
                              'space_x': np.random.randint(low=4, high=21),
                              'radius': np.random.randint(low=1, high=2)}
        
        elif sparsity_mode == 'scribbles':
            
            # Randomly choosing scribble proportion, radius and thickness; number of slices and axis.
            sparsity_param = {'num_slices': 1, #np.random.randint(low=1, high=6),
                              'axis': np.random.randint(low=0, high=3),
                              'prop': np.random.random(),
                              'dist': np.random.randint(low=2, high=6),
                              'thick': np.random.randint(low=1, high=4)}
        
        elif sparsity_mode == 'contours':
            
            # Randomly choosing contour proportion, radius and thickness; number of slices and axis.
            sparsity_param = {'num_slices': 1, #np.random.randint(low=1, high=6),
                              'axis': np.random.randint(low=0, high=3),
                              'prop': np.random.random(),
                              'dist': np.random.randint(low=2, high=6),
                              'thick': np.random.randint(low=0, high=2)}
        
        elif sparsity_mode == 'skeleton':
            
            # Randomly choosing skeleton radius and thickness; number of slices and axis.
            sparsity_param = {'num_slices': 1, #np.random.randint(low=1, high=6),
                              'axis': np.random.randint(low=0, high=3),
                              'dist': np.random.randint(low=2, high=8),
                              'thick': np.random.randint(low=0, high=2)}
        
        elif sparsity_mode == 'polygons':
            
            # Randomly choosing polygon radius and tolerance; number of slices and axis.
            sparsity_param = {'num_slices': 1, #np.random.randint(low=1, high=6),
                              'axis': np.random.randint(low=0, high=3),
                              'dist': np.random.randint(low=2, high=6),
                              'tol_neg': (np.random.random() * 20.0) + 20.0,
                              'tol_pos': (np.random.random() * 10.0) + 5.0}
        
        elif sparsity_mode == 'slices' or sparsity_mode == 'dense':
            
            # Randomly choosing number of slices and axis.
            sparsity_param = {'num_slices': 1, #np.random.randint(low=1, high=6),
                              'axis': np.random.randint(low=0, high=3)}
        
        if self.verbose:
            print('Mode', sparsity_mode)
            print('Params', sparsity_param)
            logging.info('Mode ' + str(sparsity_mode))
            logging.info('Params ' + str(sparsity_param))
        
        # Initiating support and query bundles.
        print(tsk_dict['dataset'], tsk_dict['task'])
        curr_bundle = {'task': tsk_dict['dataset'] + ' ' + tsk_dict['task']['name'],
                       'mode': sparsity_mode,
                       'sup': []}
        
        # Volume names.
        sup_names = []
        qry_names = []
        
        # Iterating over support samples.
        for sup_i, ps in enumerate(perm_sup):
            
            # Obtaining paths for images and masks.
            img_path, msk_path = tsk_dict['sup'][ps]
            
            sup_names.append(img_path.split('/')[-1])
            
            # Reading samples.
            img, msk, sparse_msk, bundle = self.read_sample(img_path, msk_path, src_indices, trg_index, augmentation, sparsity_mode, sparsity_param, seed)
            
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
            img, msk, sparse_msk, bundle = self.read_sample(img_path, msk_path, src_indices, trg_index, augmentation, 'dense', sparsity_param, seed)
            
            # Filling tensors.
            qry_img[qry_i, 0] = img
            qry_msk[qry_i] = msk
        
        return sup_img, sup_msk, sup_dns, qry_img, qry_msk, curr_bundle, sup_names, qry_names

    def __len__(self):

        return len(self.tsks)
