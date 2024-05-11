# Meta-learners for few-shot weakly-supervised medical image segmentation

> Repository with the code for conducting the experiments of the paper ** Meta-learners for few-shot weakly-supervised medical image segmentation **

[[paper]](https://www.sciencedirect.com/science/article/pii/S003132032400222X)

Most uses of Meta-Learning in visual recognition are very often applied to image classification, with a relative lack of work in other tasks such as segmentation and detection. We propose a new generic Meta-Learning framework for few-shot weakly supervised segmentation in medical imaging domains. The proposed approach includes a meta-training phase that uses a meta-dataset. 

Our experiments consider in total 9 meta-learners, 4 backbones, and multiple target organ segmentation tasks. We explore small-data scenarios in radiology with varying weak annotation styles and densities. Our analysis shows that metric-based meta-learning approaches achieve better segmentation results in tasks with smaller domain shifts compared to the meta-training datasets, while some gradient- and fusion-based meta-learners are more generalizable to larger domain shifts.

The code in mainly built using the [PyTorch](https://pytorch.org/) deep learning library and the [learn2learn](https://learn2learn.net/) module.

### Requirements
The exact versions of PyTorch and learn2learn used were:
```
learn2learn== x.x.x
torch== x.x.x
```
Installing them with pip should download their correct dependencies.
Newer versions of both libraries will probably also work.

## Repository content and organization

- The `train_X.py`/`test_X.py` files are the main scripts to train and test a model with the meta-learning algorithm X.
- The `models` folder contains the implementation of the 4 types of network used as backbones.
- `data`  directory contains implementations of the datasets classes used in the experiments, both 2D and 3D datasets.
- `utils` directory contains utility methods, such as loss functions, experiment generators, saving functions, etc.

## Getting Started

> IMPORTANT: Before starting make sure to download the datasets and assert that they follow the structure below:

### Dataset folder structure

#### 2D datasets
Your dataset fold structure should look as below:

``` bash
root                              # root folder containing all datasets
├── jsrt                          # the primary folder of a dataset have its name (see MetaDataset.dataset_list, and change it to match the datasets in your root folder)
│   ├── imgs                      # folder with all images
│   ├── groundtruths              # folder with the masks for the images divide by task subfolders (e.g. task 1, contains the masks of the binary labels for task 1)
│   │   ├── task1                   # subfolder with task1 masks
│   │   ├── task2
│   │   ...
│   │   ├── task n
│   ├──valid_labels.txt           # text file with the name of the tasks (one per line)
│   ├── folds                     # folder with text files, where each file contains a list of the images either in the training or test set for fold k for the n-th task
│   │   ├── trn_task1_f1.txt        # text file with the list of images in training in fold 1 for task 1
│   │   ├── tst_task1_f1.txt        # text file with the list of images in test in fold 1 for task 1
│   │   ├── trn_task1_f2.txt
│   │   ├── tst_task1_f2.txt
│   │   ...
│   │   ├── trn_taskn_fk.txt
│   │   └── tst_taskn_fk.txt
├── montgomery                    # other named datasets, that have the same subfolder organization as above
├── shenzhen
├── openist
...
└── panoramic
```

#### 3D datasets
Your dataset fold structure should look as below:

``` bash
root                              # root folder containing all datasets
├── dataset1                      # the primary folder of a dataset have its name (see MetaDataset3D.dataset_list, and change it to match the datasets in your root folder)
│   ├── imgs                      # folder with all images
│   ├── groundtruths              # folder with the binary masks for the images
│   ├── valid_labels.txt          # text file with the name of the task (one line)
│   ├── folds                     # folder with text files, where each file contains a list of the images either in the training or test set for fold k
│   │   ├── trn_f1.txt
│   │   ├── tst_f1.txt
│   │   ├── trn_f2.txt
│   │   ├── tst_f2.txt
│   │   ...
│   │   ├── trn_fk.txt
│   │   ├── tst_fk.txt
├── dataset2                      # other named datasets, that have the same subfolder organization as above
├── dataset3
...
└── datasetM
```

*Here `k` represents the total number of folds you selected to divide the dataset. In our experiments we used `k=5`.*

## Examples of Use

Assuming that all datasets are downloaded and organized as above, and are all located in a root folder `..\Datasets`.
To train a model using one of the meta-learning methods, one can direct use the script associated with said method. For example, to train a model using the ANIL algorithm:
```bash
# Train a ResNet18 FCN model using the ANIL and Selective Cross-Entropy loss, for the few-shot task of segmenting lungs in the JSRT dataset
python train_anil.py jsrt lungs resnet18 sce
```

To test the trained model above, you can execute:
```bash
# Train a ResNet18 FCN model using the ANIL and Selective Cross-Entropy loss, for the few-shot task of segmenting lungs in the JSRT dataset
python test_anil.py jsrt lungs resnet18 sce
```

All scripts use the same four positional arguments in either training or testing, they are:

### Key Parameters:
| **Position** 	| **Description**                                                                                                                          	|
|--------------	|------------------------------------------------------------------------------------------------------------------------------------------	|
| 1            	| Name of the target dataset. (e.g. jsrt, panoramic, mias, etc.)                                                                           	|
| 2            	| Name of the target task. (e.g. lungs, mandibule, breast)                                                                                 	|
| 3            	| Name of the network model. (Options: unet, efficientlab, deeplabv3, resnet12\|18\|50 )                                                   	|
| 4            	| Type of loss used in training (Options: sce [Selective Cross Entropy], dice, sce+dice [a combination of both losses], focal [Focal Loss])	|

*An additional optional positional parameter is available only during training of the anil, panet, metaoptnet_ridge, and r2d2. This additional parameter is the path of a pretrained model of a Resnet18 or Resnet50 FCN using SSL. These weights can be downloaded [here](#)*

### Available meta-learners
Gradient-Based:
- 'anil': Rapid learning or feature reuse? towards understanding the effectiveness of maml [[paper]](https://arxiv.org/pdf/1909.09157)
- 'maml': Model-agnostic meta-learning for fast adaptation of deep networks [[paper]](http://proceedings.mlr.press/v70/finn17a/finn17a.pdf)
- 'metasgd': Meta-sgd: Learning to learn quickly for few-shot learning [[paper]](https://arxiv.org/pdf/1707.09835)
- 'reptile': Reptile: a scalable metalearning algorithm [[paper]](https://yobibyte.github.io/files/paper_notes/Reptile___a_Scalable_Metalearning_Algorithm__Alex_Nichol_and_John_Schulman__2018.pdf)

Metric-Based:
- 'protonet': Prototypical networks for few-shot learning [[paper]](https://proceedings.neurips.cc/paper_files/paper/2017/file/cb8da6767461f2812ae4290eac7cbc42-Paper.pdf)
- 'panet': Panet: Few-shot image semantic segmentation with prototype alignment [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_PANet_Few-Shot_Image_Semantic_Segmentation_With_Prototype_Alignment_ICCV_2019_paper.pdf)

Fusion-Based:
- 'guided_net': Few-shot segmentation propagation with guided networks [[paper]](https://arxiv.org/pdf/1806.07373)
- 'metaoptnet_ridge': Meta-learning with differentiable convex optimization [[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lee_Meta-Learning_With_Differentiable_Convex_Optimization_CVPR_2019_paper.pdf)
- 'r2d2': Meta-learning with differentiable closed-form solvers  [[paper]](https://arxiv.org/pdf/1805.08136)

For testing only:
- 'baseline': A simple fine-tuning in the target task.

## Datasets
### 2d:
The full list of datasets used are:

- jsrt: JSRT Database [[source]](https://ajronline.org/doi/full/10.2214/ajr.174.1.1740071)
- montgomery: Montgomery Dataset [[source]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/)
- shenzhen: Shenzhen Dataset [[source]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/)
- openist: OpenIST Chest X-Rays dataset [[source]](https://github.com/pi-null-mezon/OpenIST)
- nih_labeled: NIH-labeled dataset [[source]](http://proceedings.mlr.press/v102/tang19a/tang19a.pdf)
- mias: MIAS database [[source]](https://www.repository.cam.ac.uk/items/b6a97f0c-3b9b-40ad-8f18-3d121eef1459)
- inbreast: INbreast database [[source]](https://repositorio.inesctec.pt/bitstream/123456789/2296/1/PS-07363.pdf)
- ivisionlab: IVisionLab Dental Images Dataset [[source]](https://arxiv.org/pdf/1802.03086)
- panoramic: Panoramic Dental X-rays [[source]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4652330/)

### 3d:
Some of the datasets used:
- structseg_head, structseg_thorax: StructSeg (A large annotated medical image dataset for the development and evaluation of segmentation algorithms) [[paper]](https://arxiv.org/pdf/1902.09063)
- decathlon_liver, decathlon_pancreas, decathlon_spleen: MSD (The medical segmentation decathlon) [[paper]](https://www.nature.com/articles/s41467-022-30695-9)

# Cite us:
If you have found our code and data useful, we kindly ask you to cite our work:
```tex
@article{oliveira2024meta,
title = {Meta-learners for few-shot weakly-supervised medical image segmentation},
journal = {Pattern Recognition},
volume = {153},
pages = {110471},
year = {2024},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2024.110471},
url = {https://www.sciencedirect.com/science/article/pii/S003132032400222X},
author = {Hugo Oliveira and Pedro H.T. Gama and Isabelle Bloch and Roberto Marcondes Cesar},
keywords = {Meta-learning, Weakly supervised segmentation, Few-shot learning, Medical images, Domain generalization},
abstract = {Most uses of Meta-Learning in visual recognition are very often applied to image classification, with a relative lack of work in other tasks such as segmentation and detection. We propose a new generic Meta-Learning framework for few-shot weakly supervised segmentation in medical imaging domains. The proposed approach includes a meta-training phase that uses a meta-dataset. It is deployed on an out-of-distribution few-shot target task, where a single highly generalizable model, trained via a selective supervised loss function, is used as a predictor. The model can be trained in several distinct ways, such as second-order optimization, metric learning, and late fusion. Some relevant improvements of existing methods that are part of the proposed approach are presented. We conduct a comparative analysis of meta-learners from distinct paradigms adapted to few-shot image segmentation in different sparsely annotated radiological tasks. The imaging modalities include 2D chest, mammographic, and dental X-rays, as well as 2D slices of volumetric tomography and resonance images. Our experiments consider in total 9 meta-learners, 4 backbones, and multiple target organ segmentation tasks. We explore small-data scenarios in radiology with varying weak annotation styles and densities. Our analysis shows that metric-based meta-learning approaches achieve better segmentation results in tasks with smaller domain shifts compared to the meta-training datasets, while some gradient- and fusion-based meta-learners are more generalizable to larger domain shifts. Guidelines learned from the comparative performance assessment of the analyzed methods are summarized to support those readers interested in the field.}
}
```
The paper is also available in the Pattern Recognition Journal, found online in: [https://www.sciencedirect.com/science/article/pii/S003132032400222X](https://www.sciencedirect.com/science/article/pii/S003132032400222X)


<!---
# Models:
Foi usado apenas pretreino SSL. Remover: 
    - Attention
    - Aggregate
    - CRF

    Unet [x]
    EfficientLab [x]
    DeepLab [x]
    Resnet12 [x]
    Resnet18 [x]
    Resnet50 [?] (checar o load de pretrained depois (é uma resnet50 qualquer ou um FCNResnet50?))

# Meta Learners
Foi usado apenas pretreino SSL. Remover: 
    - Attention
    - Aggregate
    - CRF

    Baseline [x]
    MAML [x]
    ANIL [x]
    Reptile [x]
    MetaSGD [x]

    ProtoNets [x]
    Panets [x]

    MetaOptNet [x]
    R2D2 [x]
    Guided Net [x]

    Classification Heads [x]

# Dataset related
    Meta Dataset [x]
    Meta Dataset 3d [x]

    Query Dataset [x]
    Query Dataset 3d [x]

    Support Dataset [x]
    Support Dataset 3d [x]

    Sparsify [x]
    Sparsify 3d [x]

# Utils
Apenas experimentos 2d e 3d. Sem iterativos
    Utils [x]
    Experiments [x]
    Losses [x]

--->