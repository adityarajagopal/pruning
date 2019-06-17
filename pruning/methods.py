import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from collections import Counter

from pruning.utils import prune_rate, arg_nonzero_min

def weight_prune(params, model, pruning_perc):
    '''
    Prune pruning_perc% weights globally (not layer-wise)
    arXiv: 1606.09274
    '''    
    all_weights = []
    layerCount = 0
    for p in model.named_parameters():
        if 'conv' in p[0] and 'weight' in p[0]:
            if layerCount >= params.thisLayerUp:
                all_weights += list(p[1].cpu().data.abs().numpy().flatten())
            layerCount += 1
    threshold = np.percentile(np.array(all_weights), pruning_perc)

    # generate mask
    masks = []
    layer_count = 0
    for p in model.named_parameters():
       if 'conv' in p[0] and 'weight' in p[0]:
           if layer_count >= params.thisLayerUp :
               pruned_inds = p[1].data.abs() > threshold
               mask = pruned_inds.float()
           else :
               mask = torch.tensor((), dtype=torch.float32)
               mask = mask.new_ones(p[1].size())

           layer_count += 1            
           
           masks.append(mask)
    
    return masks

def prune_one_filter(params, model, masks):
    '''
    Pruning one least ``important'' feature map by the scaled l2norm of 
    kernel weights
    arXiv:1611.06440
    '''
    NO_MASKS = False
    # construct masks if there is not yet
    if not masks:
        masks = []
        NO_MASKS = True

    values = []
    for p in model.named_parameters():
        if 'conv' in p[0] and 'weight' in p[0]: 
            p_np = p[1].data.cpu().numpy()

            # construct masks if there is not
            if NO_MASKS:
                tmp = torch.tensor((), dtype=torch.float32)
                masks.append(tmp.new_ones(p_np.shape))

            # find the scaled l2 norm for each filter this layer
            value_this_layer = np.square(p_np).sum(axis=1).sum(axis=1)\
                .sum(axis=1)/(p_np.shape[1]*p_np.shape[2]*p_np.shape[3])
            
            # normalization (important)
            value_this_layer = value_this_layer / \
                np.sqrt(np.square(value_this_layer).sum())
            
            min_value, min_ind = arg_nonzero_min(list(value_this_layer))
            
            values.append([min_value, min_ind])

    assert len(masks) == len(values), "something wrong here"

    values = np.array(values)

    # set mask corresponding to the filter to prune
    to_prune_layer_ind = np.argmin(values[params.thisLayerUp:, 0])
    to_prune_layer_ind += params.thisLayerUp
    to_prune_filter_ind = int(values[to_prune_layer_ind, 1])
    masks[to_prune_layer_ind][to_prune_filter_ind] = 0.

    return masks

def filter_prune(params, model):
    '''
    Prune filters one by one until reach pruning_perc
    (not iterative pruning)
    '''
    masks = []
    current_pruning_perc = 0.
    # params.prunedLayers = []

    while current_pruning_perc < params.pruningPerc:
        masks = prune_one_filter(params, model, masks)
        model.module.set_masks(masks)
        current_pruning_perc = prune_rate(params, model, verbose=False)
    
    return masks

def prune_model(params, model) : 
    if params.pruneWeights == True: 
        masks = weight_prune(params, model, params.pruningPerc)
        model.module.set_masks(masks)
    
    elif params.pruneFilters == True: 
        masks = filter_prune(params, model)

    return model
