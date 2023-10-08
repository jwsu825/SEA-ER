# SEA-ER

# implementation for the paper "ON LEARNABILITY AND EXPERIENCE REPLAY METHODS FOR GRAPH INCREMENTAL LEARNING ON EVOLVING GRAPHS"


 ## Get Started
 
 This repository is implemented for GPU devices and is adopted from the [CGLB repo](https://github.com/QueuQ/CGLB). 
 The requirements and steps to run the repo can be found in [CGLB repo](https://github.com/QueuQ/CGLB) with additional requirement for cvxopt library.

##  Instruction 

### run ergnn of selected sampling method with importance reweighting


 ```
 python train.py --dataset Arxiv-CL \
        --method ergnn \
        --backbone GCN \
        --gpu 0 \
        --ergnn selected_sampling_method \
        --ir True\
 ```
