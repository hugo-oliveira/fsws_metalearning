#!/bin/bash

# $1: is either train or test, i.e. if you want to train or evaluate a meta-learning method
# $2: is the name of the meta-learning algorithm. Possible values: 
#                                                 - Grad: [anil, maml, metasgd, reptile] 
#                                                 - Metric: [protonet, panet]
#                                                 - Fusion: [guided_net, metaoptnet_ridge, r2d2]
#                                                 - For test only: baseline (a simple finetuning)
# $3: is the name of the target dataset
# $4: is the target task
# $5: is the segmentation network model [unet, efficientlab, deeplabv3, resnet12|18|50, ]
# $6: is the loss function [sce, dice, sce+dice, focal]
# $7: path to pretrained weights for a resnet18 or 50. ONLY AVAILABLE for the methods anil, panet, metaoptnet_ridge, and r2d2

python $1_$2.py $3 $4 $5 $6 $7