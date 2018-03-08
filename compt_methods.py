import sktensor
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
        U,sigma,VT = np.linalg.svd(B)
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
def tucker_cpt(sparse_data,miss_loc,rank_list,W):
    est_data = sparse_data.copy()
    dshape = np.shape(est_data)
    SD = dtensor(est_data)
    U = tucker.hosvd(SD,rank_list)
    core1,U1 = tucker.hooi(SD,rank_list,init='nvecs')
    #ttm:矩阵乘法
    ttm_data = core1.ttm(U1[0],0).ttm(U1[1],1).ttm(U1[2],2)
    print(np.sum((est_data*W-ttm_data*W)**2)/np.sum((est_data*W)**2))
    miss_sum = 0
    for _set in miss_loc:
        i,j,k = _set
        est_data[i,j,k] = ttm_data[i,j,k]
        miss_sum += ttm_data[i,j,k]
    return est_data

#STD填充
def STD_cpt(sparse_data,miss_loc,U_dict,r_dict,W):
    return 0

#CP分解的填充方法
def cp_cpt(sparse_data,miss_loc,rank):
    est_data = sparse_data.copy()
    dshape = np.shape(est_data)
    SD = dtensor(sparse_data.copy())
    U = []
    P,fit,itr,arr = cp.als(SD,rank)
    loc_data = P.totensor()
    for _set in miss_loc:
        i,j,k = _set
        est_data[i,j,k] = loc_data[i,j,k]
    return est_data

#LRTC的填充方法
def lrtc_cpt(sparse_data,miss_loc,alpha,beta,gama,conv_thre,K,W):
    Y = sparse_data.copy()
    N = len(np.shape(sparse_data))
    for _set in miss_loc:
        i,j,k = _set
        #Y[i,j,k] = 0
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
            para_fi = gama[n]/(alpha[n]+beta[n])
            U,sigma,VT = np.linalg.svd(M_temp)
            row_s = len(sigma)
            mat_sig = np.zeros((row_s,row_s))
            max_rank = 0
            for ii in range(row_s):
                mat_sig[ii,ii] = max(sigma[ii]-para_fi,0)
            M[n] = np.dot(np.dot(U[:,:row_s],mat_sig),VT[:row_s,:])
            M_fold[n] = M[n].fold()
        X = np.sum([alpha[i]*M_fold[i] for i in range(N)],axis=0)/sum(alpha)
        Y_temp = np.sum([beta[i]*M_fold[i] for i in range(N)],axis=0)/sum(beta)
        for _set in miss_loc:
            p,q,r = _set
            Y[p,q,r] = Y_temp[p,q,r]
        Y_Fnorm = np.sum((Y-Y_pre)**2)
        if Y_Fnorm < conv_thre:
            print(Y_Fnorm)
            break
    print('iter:',iter,Y_Fnorm)
    return Y
 
def silrtc_cpt(sparse_data,miss_loc,alpha,beta,conv_thre,K):
    X = sparse_data.copy()
    for _set in miss_loc:
        p,q,r = _set
        #X[p,q,r] = 0
    M = {}
    N = len(np.shape(X))
    for iter in range(K):
        X_pre = X.copy()
        for i in range(N):
            para_fi = alpha[i]/beta[i]
            U,sigma,VT = np.linalg.svd(dtensor(X).unfold(i))
            row_s = len(sigma)
            mat_sig = np.zeros((row_s,row_s))
            for ii in range(row_s):
                mat_sig[ii,ii] = max(sigma[ii]-para_fi,0)
            M[i] = (np.dot(np.dot(U,mat_sig),VT[:row_s,:])).fold()
        X_temp = np.sum([beta[j]*M[j] for j in range(N)],axis=0)/sum(beta)
        for _set in miss_loc:
            p,q,r = _set
            X[p,q,r] = X_temp[p,q,r]
        X_diffnorm = np.sum((X-X_pre)**2)
        if X_diffnorm < conv_thre:
            break
    print(iter,X_diffnorm)
    return X

def halrtc_cpt(sparse_data,miss_loc,lou,conv_thre,K,W):
    X = sparse_data.copy()
    Y = {}
    N = len(np.shape(X))
    W1 = (W==False)
    M = {}
    alpha = np.array([1/N]).repeat(N)
    SD = dtensor(X)
    for _ in range(N):
        Y[_] = dtensor(np.zeros(np.shape(X)))
    for _set in miss_loc:
        p,q,r = _set
        #X[p,q,r] = 0
    for iter in range(K):
        X_pre = X.copy()
        for i in range(N):
            Matrix = SD.unfold(i)+1/lou*Y[i].unfold(i)
            U,sigma,VT = np.linalg.svd(Matrix)  
            row_s = len(sigma)
            mat_sig = np.zeros((row_s,row_s))
            for ii in range(row_s):
                mat_sig[ii,ii] = max(sigma[ii]-alpha[i]/lou,0)
            M[i] = (np.dot(np.dot(U,mat_sig),VT[:row_s,:])).fold()
        T_temp = (np.sum([M[j]-1/lou*Y[i] for j in range(N)],axis=0))/N
        X = X*W+T_temp*W1
        X_Fnorm = np.sum((X-X_pre)**2)
        if X_Fnorm < conv_thre:
            break
        for i in range(N):
            Y[i] -= lou*(M[i]-X)
    print(iter,X_Fnorm)
    return X
