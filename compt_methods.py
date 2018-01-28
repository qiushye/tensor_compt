import sktensor
#import os
#import scipy.io as scio
import numpy as np
#from random import random
from sktensor import tucker
from scipy.sparse import rand as sprand
from sktensor.dtensor import dtensor, unfolded_dtensor
from sktensor.sptensor import unfolded_sptensor


def tucker_cpt(ori_data,sparse_data,miss_loc,rank_list):
    data = ori_data.copy()
    sdata = sparse_data.copy()
    dshape = np.shape(data)
    SD = dtensor(sparse_data)
    core1,U1 = tucker.hooi(SD,rank_list,init='nvecs')
    #ttm:¾ØÕó³Ë·¨
    ttm_data = core1.ttm(U1[0],0).ttm(U1[1],1).ttm(U1[2],2)
    est_data = ori_data.copy()
    miss_sum = 0
    for _set in miss_loc:
        i,j,k = _set
        est_data[i,j,k] = ttm_data[i,j,k]
        miss_sum += ttm_data[i,j,k]
    print(miss_sum/len(miss_loc))
    return est_data

def cp_cpt(ori_data,sparse_data,miss_loc,rank):
    data = ori_data.copy()
    sdata = sparse_data.copy()
    dshape = np.shape(data)
    SD = dtensor(sparse_data)


