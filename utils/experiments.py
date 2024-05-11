from data.support_dataset import *
from data.query_dataset import *

from data.support_dataset_3d import *
from data.query_dataset_3d import *

from torch.utils import data 

###################################################
# 2D experiments. #################################
###################################################

def pnts_experiments(root, fold, dataset, task, resize_to, normalization, shots_list):
    
    num_points = [1, 5, 10, 20] # Number of labeled points of each class per support sample.
    radius = [1, 2, 3]          # Radius of the dilation over the points.
    
    sup_list = []
    
    # Iterating over params.
    for s in shots_list:
        for n in num_points:
            for r in radius:
                
                sparsity_param = {'num_points': n,
                                  'radius': r}
                
                # Setting sparsely labeled support dataset.
                sup_dataset = SupportDataset(root,
                                             fold,
                                             dataset,
                                             task,
                                             'points',
                                             sparsity_param,
                                             resize_to=resize_to,
                                             normalization=normalization,
                                             num_shots=s,
                                             debug=False,
                                             verbose=False)
                
                experiment = {'shots': s,
                              'sparsity': sparsity_param,
                              'sup_dataset': sup_dataset}
                
                sup_list.append(experiment)
                
    # Setting query dataset.
    qry_dataset = QueryDataset(root,
                               fold,
                               dataset,
                               task,
                               resize_to=resize_to,
                               normalization=normalization,
                               debug=False,
                               verbose=False)
    
    # Query dataloader.
    qry_loader = data.DataLoader(qry_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 drop_last=False)
    
    return sup_list, qry_loader
    
def grid_experiments(root, fold, dataset, task, resize_to, normalization, shots_list):
    
    space = [8, 12, 16, 20] # Spacings between annotations at x and y axes.
    radius = [1, 2, 3]         # Radius of the dilation over the points.
    
    sup_list = []
    
    # Iterating over params.
    for s in shots_list:
        for sp in space:
            for r in radius:
                
                sparsity_param = {'space_y': sp,
                                  'space_x': sp,
                                  'radius': r}
                
                # Setting sparsely labeled support dataset.
                sup_dataset = SupportDataset(root,
                                             fold,
                                             dataset,
                                             task,
                                             'grid',
                                             sparsity_param,
                                             resize_to=resize_to,
                                             normalization=normalization,
                                             num_shots=s,
                                             debug=False,
                                             verbose=False)
                
                experiment = {'shots': s,
                              'sparsity': sparsity_param,
                              'sup_dataset': sup_dataset}
                
                sup_list.append(experiment)
    
    # Setting query dataset.
    qry_dataset = QueryDataset(root,
                               fold,
                               dataset,
                               task,
                               resize_to=resize_to,
                               normalization=normalization,
                               debug=False,
                               verbose=False)
    
    # Query dataloader.
    qry_loader = data.DataLoader(qry_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 drop_last=False)
    
    return sup_list, qry_loader
    
def scrb_experiments(root, fold, dataset, task, resize_to, normalization, shots_list):
    
    prop = [0.05, 0.10, 0.25, 0.50, 1.00] # Proportion of contours that are annotated.
    dist = [2, 3, 4, 5]                   # Radius of the morph element for computing the internal and external contours.
    thick = [1, 2, 4]                     # Radius of the dilation over the points determining the thickness o the scribbles.
    
    sup_list = []
    
    # Iterating over params.
    for s in shots_list:
        for p in prop:
            for d in dist:
                for t in thick:
                    
                    sparsity_param = {'prop': p,
                                      'dist': d,
                                      'thick': t}
                    
                    # Setting sparsely labeled support dataset.
                    sup_dataset = SupportDataset(root,
                                                 fold,
                                                 dataset,
                                                 task,
                                                 'scribbles',
                                                 sparsity_param,
                                                 resize_to=resize_to,
                                                 normalization=normalization,
                                                 num_shots=s,
                                                 debug=False,
                                                 verbose=False)
                    
                    experiment = {'shots': s,
                                  'sparsity': sparsity_param,
                                  'sup_dataset': sup_dataset}
                    
                    sup_list.append(experiment)
    
    # Setting query dataset.
    qry_dataset = QueryDataset(root,
                               fold,
                               dataset,
                               task,
                               resize_to=resize_to,
                               normalization=normalization,
                               debug=False,
                               verbose=False)
    
    # Query dataloader.
    qry_loader = data.DataLoader(qry_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 drop_last=False)
    
    return sup_list, qry_loader

def cntr_experiments(root, fold, dataset, task, resize_to, normalization, shots_list):
    
    prop = [0.05, 0.10, 0.25, 0.50, 1.00] # Proportion of contours that are annotated.
    thick = [1, 2, 4, 8]                  # Thickness of the contours that are annotated.
    
    sup_list = []
    
    # Iterating over params.
    for s in shots_list:
        for p in prop:
            for t in thick:
                
                sparsity_param = {'prop': p,
                                  'thick': t}
                
                sup_dataset = SupportDataset(root,
                                             fold,
                                             dataset,
                                             task,
                                             'contours',
                                             sparsity_param,
                                             resize_to=resize_to,
                                             normalization=normalization,
                                             num_shots=s,
                                             debug=False,
                                             verbose=False)
                
                experiment = {'shots': s,
                              'sparsity': sparsity_param,
                              'sup_dataset': sup_dataset}
                
                sup_list.append(experiment)
    
    # Setting query dataset.
    qry_dataset = QueryDataset(root,
                               fold,
                               dataset,
                               task,
                               resize_to=resize_to,
                               normalization=normalization,
                               debug=False,
                               verbose=False)
    
    # Query dataloader.
    qry_loader = data.DataLoader(qry_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 drop_last=False)
    
    return sup_list, qry_loader

def skel_experiments(root, fold, dataset, task, resize_to, normalization, shots_list):
    
    dist = [2, 4, 6]  # Radius of the morph element for computing the external contours.
    thick = [1, 2, 4] # Thickness of skeleton line and external contours.
    
    sup_list = []
    
    # Iterating over params.
    for s in shots_list:
        for d in dist:
            for t in thick:
                
                sparsity_param = {'dist': d,
                                  'thick': t}
                
                # Setting sparsely labeled support dataset.
                sup_dataset = SupportDataset(root,
                                             fold,
                                             dataset,
                                             task,
                                             'skeleton',
                                             sparsity_param,
                                             resize_to=resize_to,
                                             normalization=normalization,
                                             num_shots=s,
                                             debug=False,
                                             verbose=False)
                
                experiment = {'shots': s,
                              'sparsity': sparsity_param,
                              'sup_dataset': sup_dataset}
                
                sup_list.append(experiment)
    
    # Setting query dataset.
    qry_dataset = QueryDataset(root,
                               fold,
                               dataset,
                               task,
                               resize_to=resize_to,
                               normalization=normalization,
                               debug=False,
                               verbose=False)
    
    # Query dataloader.
    qry_loader = data.DataLoader(qry_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 drop_last=False)
    
    return sup_list, qry_loader

def poly_experiments(root, fold, dataset, task, resize_to, normalization, shots_list):
    
    dist = [2, 3, 4, 5]    # Radius of the morph element for dilating the convex hull.
    tol_neg = [20, 30, 40] # Tolerance for the negative class in polygon approximation.
    tol_pos = [5, 10, 15]  # Tolerance for the positive class in polygon approximation.
    
    sup_list = []
    
    # Iterating over params.
    for s in shots_list:
        for d in dist:
            for tn in tol_neg:
                for tp in tol_pos:
                    
                    sparsity_param = {'dist': d,
                                      'tol_neg': tn,
                                      'tol_pos': tp}
                    
                    # Setting sparsely labeled support dataset.
                    sup_dataset = SupportDataset(root,
                                                 fold,
                                                 dataset,
                                                 task,
                                                 'polygons',
                                                 sparsity_param,
                                                 resize_to=resize_to,
                                                 normalization=normalization,
                                                 num_shots=s,
                                                 debug=False,
                                                 verbose=False)
                    
                    experiment = {'shots': s,
                                  'sparsity': sparsity_param,
                                  'sup_dataset': sup_dataset}
                    
                    sup_list.append(experiment)
    
    # Setting query dataset.
    qry_dataset = QueryDataset(root,
                               fold,
                               dataset,
                               task,
                               resize_to=resize_to,
                               normalization=normalization,
                               debug=False,
                               verbose=False)
    
    # Query dataloader.
    qry_loader = data.DataLoader(qry_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 drop_last=False)
    
    return sup_list, qry_loader

def dens_experiments(root, fold, dataset, task, resize_to, normalization, shots_list):
    
    sup_list = []
    
    # Iterating over params.
    for s in shots_list:
        
        sparsity_param = {}
        
        # Setting sparsely labeled support dataset.
        sup_dataset = SupportDataset(root,
                                     fold,
                                     dataset,
                                     task,
                                     'dense',
                                     sparsity_param,
                                     resize_to=resize_to,
                                     normalization=normalization,
                                     num_shots=s,
                                     debug=False,
                                     verbose=False)
        
        experiment = {'shots': s,
                      'sparsity': sparsity_param,
                      'sup_dataset': sup_dataset}
        
        sup_list.append(experiment)
    
    # Setting query dataset.
    qry_dataset = QueryDataset(root,
                               fold,
                               dataset,
                               task,
                               resize_to=resize_to,
                               normalization=normalization,
                               debug=False,
                               verbose=False)
    
    # Query dataloader.
    qry_loader = data.DataLoader(qry_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 drop_last=False)
    
    return sup_list, qry_loader

def sup_qry_experiments(root, fold, dataset, task, resize_to, normalization, shots_list):
    
    # Obtaining sparse label experiments composed of support and query datasets.
    exp_pnts = pnts_experiments(root, fold, dataset, task, resize_to, normalization, shots_list)
    exp_grid = grid_experiments(root, fold, dataset, task, resize_to, normalization, shots_list)
    exp_scrb = scrb_experiments(root, fold, dataset, task, resize_to, normalization, shots_list)
    exp_cntr = cntr_experiments(root, fold, dataset, task, resize_to, normalization, shots_list)
    exp_skel = skel_experiments(root, fold, dataset, task, resize_to, normalization, shots_list)
    exp_poly = poly_experiments(root, fold, dataset, task, resize_to, normalization, shots_list)
    exp_dens = dens_experiments(root, fold, dataset, task, resize_to, normalization, shots_list)
    
    return exp_pnts, exp_grid, exp_scrb, exp_cntr, exp_skel, exp_poly, exp_dens


###################################################
# 3D slice experiments. ###########################
###################################################

def pnts_experiments_3d(root, fold, dataset, task, resize_to, normalization, shots_list):
    
    num_points = [1, 5, 10] # Number of labeled points of each class per support sample.
    radius = [1, 3]         # Radius of the dilation over the points.
    
    sup_list = []
    
    # Iterating over params.
    for s in shots_list:
        for n in num_points:
            for r in radius:
                
                sparsity_param = {'num_slices': 1, #np.random.randint(low=1, high=6),
                                  'axis': 2,
                                  'num_points': n,
                                  'radius': r}
                
                # Setting sparsely labeled support dataset.
                sup_dataset = SupportDataset3D(root,
                                               fold,
                                               dataset,
                                               task,
                                               'points',
                                               sparsity_param,
                                               resize_to=resize_to,
                                               normalization=normalization,
                                               num_shots=s,
                                               debug=False,
                                               verbose=False)
                
                experiment = {'shots': s,
                              'sparsity': sparsity_param,
                              'sup_dataset': sup_dataset}
                
                sup_list.append(experiment)
                
    # Setting query dataset.
    qry_dataset = QueryDataset3D(root,
                                 fold,
                                 dataset,
                                 task,
                                 axis=2,
                                 resize_to=resize_to,
                                 normalization=normalization,
                                 debug=False,
                                 verbose=False)
    
    # Query dataloader.
    qry_loader = data.DataLoader(qry_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=5,
                                 drop_last=False)
    
    return sup_list, qry_loader
    
def grid_experiments_3d(root, fold, dataset, task, resize_to, normalization, shots_list):
    
    space = [12, 16, 20] # Spacings between annotations at x and y axes.
    radius = [1, 2, 3]   # Radius of the dilation over the points.
    
    sup_list = []
    
    # Iterating over params.
    for s in shots_list:
        for sp in space:
            for r in radius:
                
                sparsity_param = {'num_slices': 1, #np.random.randint(low=1, high=6),
                                  'axis': 2,
                                  'space_y': sp,
                                  'space_x': sp,
                                  'radius': r}
                
                # Setting sparsely labeled support dataset.
                sup_dataset = SupportDataset3D(root,
                                               fold,
                                               dataset,
                                               task,
                                               'grid',
                                               sparsity_param,
                                               resize_to=resize_to,
                                               normalization=normalization,
                                               num_shots=s,
                                               debug=False,
                                               verbose=False)
                
                experiment = {'shots': s,
                              'sparsity': sparsity_param,
                              'sup_dataset': sup_dataset}
                
                sup_list.append(experiment)
    
    # Setting query dataset.
    qry_dataset = QueryDataset3D(root,
                                 fold,
                                 dataset,
                                 task,
                                 axis=2,
                                 resize_to=resize_to,
                                 normalization=normalization,
                                 debug=False,
                                 verbose=False)
    
    # Query dataloader.
    qry_loader = data.DataLoader(qry_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=5,
                                 drop_last=False)
    
    return sup_list, qry_loader
    
def scrb_experiments_3d(root, fold, dataset, task, resize_to, normalization, shots_list):
    
    prop = [0.05, 0.10, 0.25, 0.50, 1.00] # Proportion of contours that are annotated.
    dist = [2, 4, 6, 8]                   # Radius of the morph element for computing the internal and external contours.
    thick = [1, 2, 3, 4]                  # Radius of the dilation over the points determining the thickness o the scribbles.
    
    sup_list = []
    
    # Iterating over params.
    for s in shots_list:
        for p in prop:
            for d in dist:
                for t in thick:
                    
                    sparsity_param = {'num_slices': 1, #np.random.randint(low=1, high=6),
                                      'axis': 2,
                                      'prop': p,
                                      'dist': d,
                                      'thick': t}
                    
                    # Setting sparsely labeled support dataset.
                    sup_dataset = SupportDataset3D(root,
                                                   fold,
                                                   dataset,
                                                   task,
                                                   'scribbles',
                                                   sparsity_param,
                                                   resize_to=resize_to,
                                                   normalization=normalization,
                                                   num_shots=s,
                                                   debug=False,
                                                   verbose=False)
                    
                    experiment = {'shots': s,
                                  'sparsity': sparsity_param,
                                  'sup_dataset': sup_dataset}
                    
                    sup_list.append(experiment)
    
    # Setting query dataset.
    qry_dataset = QueryDataset3D(root,
                                 fold,
                                 dataset,
                                 task,
                                 axis=2,
                                 resize_to=resize_to,
                                 normalization=normalization,
                                 debug=False,
                                 verbose=False)
    
    # Query dataloader.
    qry_loader = data.DataLoader(qry_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=5,
                                 drop_last=False)
    
    return sup_list, qry_loader

def cntr_experiments_3d(root, fold, dataset, task, resize_to, normalization, shots_list):
    
    prop = [0.05, 0.10, 0.25, 0.50, 1.00] # Proportion of contours that are annotated.
    thick = [2, 8]                        # Thickness of the contours that are annotated.
    
    sup_list = []
    
    # Iterating over params.
    for s in shots_list:
        for p in prop:
            for t in thick:
                
                sparsity_param = {'num_slices': 1, #np.random.randint(low=1, high=6),
                                  'axis': 2,
                                  'prop': p,
                                  'thick': t}
                
                sup_dataset = SupportDataset3D(root,
                                               fold,
                                               dataset,
                                               task,
                                               'contours',
                                               sparsity_param,
                                               resize_to=resize_to,
                                               normalization=normalization,
                                               num_shots=s,
                                               debug=False,
                                               verbose=False)
                
                experiment = {'shots': s,
                              'sparsity': sparsity_param,
                              'sup_dataset': sup_dataset}
                
                sup_list.append(experiment)
    
    # Setting query dataset.
    qry_dataset = QueryDataset3D(root,
                                 fold,
                                 dataset,
                                 task,
                                 axis=2,
                                 resize_to=resize_to,
                                 normalization=normalization,
                                 debug=False,
                                 verbose=False)
    
    # Query dataloader.
    qry_loader = data.DataLoader(qry_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=5,
                                 drop_last=False)
    
    return sup_list, qry_loader

def skel_experiments_3d(root, fold, dataset, task, resize_to, normalization, shots_list):
    
    dist = [2, 4, 6]  # Radius of the morph element for computing the external contours.
    thick = [1, 2, 4] # Thickness of skeleton line and external contours.
    
    sup_list = []
    
    # Iterating over params.
    for s in shots_list:
        for d in dist:
            for t in thick:
                
                sparsity_param = {'num_slices': 1, #np.random.randint(low=1, high=6),
                                  'axis': 2,
                                  'dist': d,
                                  'thick': t}
                
                # Setting sparsely labeled support dataset.
                sup_dataset = SupportDataset3D(root,
                                               fold,
                                               dataset,
                                               task,
                                               'skeleton',
                                               sparsity_param,
                                               resize_to=resize_to,
                                               normalization=normalization,
                                               num_shots=s,
                                               debug=False,
                                               verbose=False)
                
                experiment = {'shots': s,
                              'sparsity': sparsity_param,
                              'sup_dataset': sup_dataset}
                
                sup_list.append(experiment)
    
    # Setting query dataset.
    qry_dataset = QueryDataset3D(root,
                                 fold,
                                 dataset,
                                 task,
                                 axis=2,
                                 resize_to=resize_to,
                                 normalization=normalization,
                                 debug=False,
                                 verbose=False)
    
    # Query dataloader.
    qry_loader = data.DataLoader(qry_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=5,
                                 drop_last=False)
    
    return sup_list, qry_loader

def poly_experiments_3d(root, fold, dataset, task, resize_to, normalization, shots_list):
    
    dist = [2, 5]      # Radius of the morph element for dilating the convex hull.
    tol_neg = [20, 40] # Tolerance for the negative class in polygon approximation.
    tol_pos = [5, 15]  # Tolerance for the positive class in polygon approximation.
    
    sup_list = []
    
    # Iterating over params.
    for s in shots_list:
        for d in dist:
            for tn in tol_neg:
                for tp in tol_pos:
                    
                    sparsity_param = {'num_slices': 1, #np.random.randint(low=1, high=6),
                                      'axis': 2,
                                      'dist': d,
                                      'tol_neg': tn,
                                      'tol_pos': tp}
                    
                    # Setting sparsely labeled support dataset.
                    sup_dataset = SupportDataset3D(root,
                                                   fold,
                                                   dataset,
                                                   task,
                                                   'polygons',
                                                   sparsity_param,
                                                   resize_to=resize_to,
                                                   normalization=normalization,
                                                   num_shots=s,
                                                   debug=False,
                                                   verbose=False)
                    
                    experiment = {'shots': s,
                                  'sparsity': sparsity_param,
                                  'sup_dataset': sup_dataset}
                    
                    sup_list.append(experiment)
    
    # Setting query dataset.
    qry_dataset = QueryDataset3D(root,
                                 fold,
                                 dataset,
                                 task,
                                 axis=2,
                                 resize_to=resize_to,
                                 normalization=normalization,
                                 debug=False,
                                 verbose=False)
    
    # Query dataloader.
    qry_loader = data.DataLoader(qry_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=5,
                                 drop_last=False)
    
    return sup_list, qry_loader

def dens_experiments_3d(root, fold, dataset, task, resize_to, normalization, shots_list):
    
    sup_list = []
    
    # Iterating over params.
    for s in shots_list:
        
        sparsity_param = {'num_slices': 1, #np.random.randint(low=1, high=6),
                          'axis': 2}
        
        # Setting sparsely labeled support dataset.
        sup_dataset = SupportDataset3D(root,
                                       fold,
                                       dataset,
                                       task,
                                       'dense',
                                       sparsity_param,
                                       resize_to=resize_to,
                                       normalization=normalization,
                                       num_shots=s,
                                       debug=False,
                                       verbose=False)
        
        experiment = {'shots': s,
                      'sparsity': sparsity_param,
                      'sup_dataset': sup_dataset}
        
        sup_list.append(experiment)
    
    # Setting query dataset.
    qry_dataset = QueryDataset3D(root,
                                 fold,
                                 dataset,
                                 task,
                                 2, # axis
                                 resize_to=resize_to,
                                 normalization=normalization,
                                 debug=False,
                                 verbose=False)
    
    # Query dataloader.
    qry_loader = data.DataLoader(qry_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=5,
                                 drop_last=False)
    
    return sup_list, qry_loader

def sup_qry_experiments_3d(root, fold, dataset, task, resize_to, normalization, shots_list):
    
    # Obtaining sparse label experiments composed of support and query 3D datasets.
    exp_pnts_3d = pnts_experiments_3d(root, fold, dataset, task, resize_to, normalization, shots_list)
    exp_grid_3d = grid_experiments_3d(root, fold, dataset, task, resize_to, normalization, shots_list)
    exp_scrb_3d = scrb_experiments_3d(root, fold, dataset, task, resize_to, normalization, shots_list)
    exp_cntr_3d = cntr_experiments_3d(root, fold, dataset, task, resize_to, normalization, shots_list)
    exp_skel_3d = skel_experiments_3d(root, fold, dataset, task, resize_to, normalization, shots_list)
    exp_poly_3d = poly_experiments_3d(root, fold, dataset, task, resize_to, normalization, shots_list)
    exp_dens_3d = dens_experiments_3d(root, fold, dataset, task, resize_to, normalization, shots_list)
    
    return exp_pnts_3d, exp_grid_3d, exp_scrb_3d, exp_cntr_3d, exp_skel_3d, exp_poly_3d, exp_dens_3d