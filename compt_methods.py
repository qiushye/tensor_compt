#encoding=utf-8
import scipy
import sktensor
import time
#import os
#import scipy.io as scio
import numpy as np
#from random import random
from sktensor import tucker,cp
from scipy.sparse import rand as sprand
from sktensor.dtensor import dtensor, unfolded_dtensor
from sktensor.sptensor import unfolded_sptensor


#Truncated SVD
def T_SVD(Atensor,p):
    SD = dtensor(Atensor.copy())
    N = len(np.shape(SD))
    U_dict,r_dict = {},{}
    for i in range(N):
        B = SD.unfold(i)
        U,sigma,VT = np.linalg.svd(B,0)
        row_s = len(sigma)
        mat_sig = np.zeros((row_s,row_s))
        for j in range(row_s):
            mat_sig[j,j] = sigma[j]
            if sum(sigma[:j])/sum(sigma) > p:
                break
        U_dict[i] = U[:,:j]
        r_dict[i] = j
    return U_dict,r_dict


#tucker分解的填充方法
def tucker_cpt(sparse_data,rank_list,W):
    time_s = time.time()
    est_data = sparse_data.copy()
    dshape = np.shape(est_data)
    SD = dtensor(est_data)
    #U = tucker.hosvd(SD,rank_list)
    core1,U1 = tucker.hooi(SD,rank_list,init='nvecs')
    #print('mean,var',np.mean(core1.unfold(0)),core1.var)
    #print('U_mean',(U1[0]==0).max())
    left1 = SD.unfold(0)
    U_1,sigma,VT = np.linalg.svd(left1,0)
    #print(np.sum(U1[0]-U_1))
    #ttm:矩阵乘法
    ttm_data = core1.ttm(U1[0],0).ttm(U1[1],1).ttm(U1[2],2)
    #print(np.mean(ttm_data))
    est_data = W*est_data+(W==False)*ttm_data
    time_e = time.time()
    print('-'*8+'tucker'+'-'*8)
    print('exec_time:'+str(time_e-time_s)+'s')
    return est_data

def multi_tucker(sparse_data,rates,miss_loc,W):
    SD = dtensor(sparse_data)
    est_dict = {}
    for rate in rates:
        rank_set = [0,0,0]
        for i in range(3):
            U,sigma,VT = scipy.linalg.svd(SD.unfold(i),0)
            for r in range(len(sigma)):
                if sum(sigma[:r])/sum(sigma) > rate:
                    rank_set[i] = r
                    break
        print(rank_set)
        est_dict[rate] = tucker_cpt(sparse_data,miss_loc,rank_set,W)
    return est_dict

#STD填充
def STD_cpt(sparse_data,miss_loc,U_dict,r_dict,W):
    return 0

#CP分解的填充方法
def cp_cpt(sparse_data,rank,W):
    time_s = time.time()
    est_data = sparse_data.copy()
    dshape = np.shape(est_data)
    SD = dtensor(sparse_data.copy())
    U = []
    P,fit,itr,arr = cp.als(SD,rank)
    loc_data = P.totensor()
    est_data = W*est_data+(W==False)*loc_data
    time_e = time.time()
    print('-'*8+'cp'+'-'*8)
    print('exec_time:'+str(time_e-time_s)+'s')
    return est_data

#LRTC的填充方法
def lrtc_cpt(sparse_data,alpha,beta,gama,conv_thre,K,W):
    time_s = time.time()
    Y = sparse_data.copy()
    N = len(np.shape(sparse_data))
    X = Y.copy()
    normY = np.sum(Y**2)**0.5
    M = {}
    MX,MY,M_fold = {},{},{}
    for iter in range(K):
        Y_pre = Y.copy()
        for n in range(N):
            
            MX[n] = dtensor(X).unfold(n)
            MY[n] = dtensor(Y).unfold(n)
            M_temp = (alpha[n]*MX[n]+beta[n]*MY[n])/(alpha[n]+beta[n])
            
            #U = tucker.hosvd(M_temp,M_temp.shape)
            #core,U1 = tucker.hooi(M_temp,M_temp.shape,init='nvecs')
            
            para_fi = gama[n]/(alpha[n]+beta[n])
            U,sigma,VT = np.linalg.svd(M_temp,full_matrices=0)
            row_s = len(sigma)
            mat_sig = np.zeros((row_s,row_s))
            max_rank = 0
            for ii in range(row_s):
                mat_sig[ii,ii] = max(sigma[ii]-para_fi,0)
                #mat_sig[ii,ii] = sigma[ii]
                #if sum(sigma[:ii])/sum(sigma) > 0.7:
                #    break
            M[n] = np.dot(np.dot(U[:,:row_s],mat_sig),VT[:row_s,:])
            M_fold[n] = M[n].fold()
        X = np.sum([alpha[i]*M_fold[i] for i in range(N)],axis=0)/sum(alpha)
        Y_temp = np.sum([beta[i]*M_fold[i] for i in range(N)],axis=0)/sum(beta)
        Y[W==False] = Y_temp[W==False]
        '''
        for _set in miss_loc:
            p,q,r = _set
            Y[p,q,r] = Y_temp[p,q,r]
        '''
        Y_Fnorm = np.sum((Y-Y_pre)**2)
        if Y_Fnorm < conv_thre:
            break
    time_e = time.time()
    print('-'*8+'lrtc'+'-'*8)
    print('exec_time:'+str(time_e-time_s)+'s')
    return Y
 
def silrtc_cpt(sparse_data,alpha,beta,conv_thre,K,W):
    time_s = time.time()
    X = sparse_data.copy()
    M = {}
    N = len(np.shape(X))
    for iter in range(K):
        X_pre = X.copy()
        for i in range(N):
            para_fi = alpha[i]/beta[i]
            U,sigma,VT = np.linalg.svd(dtensor(X).unfold(i),full_matrices=0)
            row_s = len(sigma)
            mat_sig = np.zeros((row_s,row_s))
            for ii in range(row_s):
                mat_sig[ii,ii] = max(sigma[ii]-para_fi,0)
                #mat_sig[ii,ii] = sigma[ii]
                #if sum(sigma[:ii])/sum(sigma) > 0.7:
                #    break
            M[i] = (np.dot(np.dot(U,mat_sig),VT[:row_s,:])).fold()
        X_temp = np.sum([beta[j]*M[j] for j in range(N)],axis=0)/sum(beta)
        X[W==False] = X_temp[W==False]
        '''
        for _set in miss_loc:
            p,q,r = _set
            X[p,q,r] = X_temp[p,q,r]
        '''
        X_diffnorm = np.sum((X-X_pre)**2)
        if X_diffnorm < conv_thre:
            break
    time_e = time.time()
    print('-'*8+'silrtc'+'-'*8)
    print('exec_time:'+str(time_e-time_s)+'s')
    return X

def halrtc_cpt(sparse_data,lou,conv_thre,K,W):
    time_s = time.time()
    X = sparse_data.copy()
    Y = {}
    N = len(np.shape(X))
    W1 = (W==False)
    M = {}
    alpha = np.array([1.0/N]).repeat(N)
    SD = dtensor(X)
    for _ in range(N):
        Y[_] = dtensor(np.zeros(np.shape(X)))
    for iter in range(K):
        X_pre = X.copy()
        for i in range(N):
            SD = dtensor(X_pre)
            Matrix = SD.unfold(i)+1/lou*(Y[i].unfold(i))
            U,sigma,VT = np.linalg.svd(Matrix,0)
            row_s = len(sigma)
            mat_sig = np.zeros((row_s,row_s))
            for ii in range(row_s):
                mat_sig[ii,ii] = max(sigma[ii]-alpha[i]/lou,0)
                #mat_sig[ii,ii] = sigma[ii]
                #if sum(sigma[:ii])/sum(sigma) > 0.7:
                #    break
            M[i] = (np.dot(np.dot(U,mat_sig),VT[:row_s,:])).fold()
        T_temp = (np.sum([M[j]-1/lou*Y[i] for j in range(N)],axis=0))/N
        X[W1] = T_temp[W1]
        X_Fnorm = np.sum((X-X_pre)**2)
        #if X_Fnorm < conv_thre:
        #    break
        for i in range(N):
            Y[i] -= lou*(M[i]-X)
    time_e = time.time()
    print('-'*8+'halrtc'+'-'*8)
    print('exec_time:'+str(time_e-time_s)+'s')
    return X
