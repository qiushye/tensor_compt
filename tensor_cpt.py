import sktensor
import sys
import os
import scipy.io as scio
import numpy as np
from random import random
from scipy.sparse import rand as sprand
from compt_methods import *

def normalize(X):
   return X / X.sum(axis=0)

def get_tensor(mat_file,ori_file,size_=0):
    ori_data = scio.loadmat(mat_file)['Speed']
    if size_:
        p,q,r = size_
        ori_data = ori_data[:p,:q,:r]
    scio.savemat(ori_file,{'Speed':ori_data})
    return ori_data

#input:normalized copied data,3-modes
def gene_sparse(ori_data, miss_radio,miss_file):
    data = ori_data.copy()
    dshape = np.shape(data)
    miss_num = np.cumprod(dshape)[-1]*miss_radio
    miss_count = 0
    for i in range(dshape[0]):
        for j in range(dshape[1]):
            for k in range(dshape[2]):
                if random() < miss_radio:
                    data[i,j,k] = 0
                    miss_count += 1
                    if miss_count == miss_num:
                        scio.savemat(miss_file,{'Speed':data})
                        return 0
    scio.savemat(miss_file,{'Speed':data})
    return 0 

def get_sparsedata(miss_file):
    miss_loc = []
    sparse_data = scio.loadmat(miss_file)['Speed']
    dshape = np.shape(sparse_data)
    for i in range(dshape[0]):
        for j in range(dshape[1]):
            for k in range(dshape[2]):
                if sparse_data[i,j,k] == 0:
                    miss_loc.append((i,j,k))
    true_miss_radio = len(miss_loc)/np.cumprod(data_size)[-1]
    return sparse_data,miss_loc,true_miss_radio

def norm_data(data_input):
    if not data_input.any():
        print('no data input')
        return data_input
    data = data_input.copy()
    mode1 = np.shape(data)[0]
    for i in range(mode1):
        norm_mode1 = dtensor(data[i,:,:]).norm()
        data[i,:,:] /= norm_mode1
    return data

def pre_impute(sparse_data,miss_loc):
    for _set in miss_loc:
        i,j,k = _set
        sparse_data[i,j,k] = np.mean(sparse_data[i,:,:])
    return sparse_data

def rmse_mape_rse(est_data,ori_data,miss_loc):
    diff_data = ori_data-est_data
    rmse = (dtensor(diff_data).norm()**2/len(miss_loc))**0.5
    mape = np.sum(np.fabs(diff_data)/ori_data)*100/len(miss_loc)
    rse = dtensor(diff_data).norm()/dtensor(ori_data).norm()
    return round(rmse,3),round(mape,3),round(rse,3)

if __name__ == '__main__':
    mat_path = '/home/qiushye/2013_east/2013_east_speed.mat'
    #数据：日期数，线圈数量，时间间隔数
    data_size = (30,20,288) 
    miss_radio = 0.1
    ori_path = 'ori_'+'_'.join([str(ch) for ch in data_size])+'.mat'
    if not os.path.exists(ori_path):
        ori_speeddata = get_tensor(mat_path,ori_path,data_size)
    else:
        ori_speeddata = scio.loadmat(ori_path)['Speed']
    miss_path = 'miss_'+'_'.join([str(ch) for ch in data_size])+'.mat'
    if not os.path.exists(miss_path):
        gene_sparse(ori_speeddata,miss_radio,miss_path)
    miss_data,miss_pos,tm_radio = get_sparsedata(miss_path)
    data_shape = np.shape(ori_speeddata)
    rank_set = [data_shape[0]//2,data_shape[1]//2,data_shape[2]//4]
    line_data = ori_speeddata[1,:,1]
    print(line_data)
    print(np.var(line_data))
    print(rank_set)
    #ori_speeddata = norm_data(ori_speeddata)
    #miss_data = norm_data(miss_data)
    miss_data = pre_impute(miss_data,miss_pos)
    print('ori_mean:',np.sum(ori_speeddata)/np.cumprod(data_size)[-1])
    est_speeddata = tucker_cpt(ori_speeddata,miss_data,miss_pos,rank_set)
    RMSE,MAPE,RSE = rmse_mape_rse(est_speeddata,ori_speeddata,miss_pos)
    print(dtensor(est_speeddata-ori_speeddata).norm()/dtensor(ori_speeddata).norm())
    print('miss_radio:',tm_radio)
    print('RMSE:',RMSE)
    print('MAPE:',MAPE)
    print('RSE:',RSE)

