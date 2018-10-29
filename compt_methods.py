#encoding=utf-8
import random
import scipy
import sktensor
import time
import sys
import os
#import scipy.io as scio
import numpy as np
#from random import random
from sktensor import tucker,cp
from scipy.sparse import rand as sprand
from sktensor.dtensor import dtensor, unfolded_dtensor
from sktensor.sptensor import unfolded_sptensor

from sklearn.cluster import KMeans
#from test_methods import deal_orimiss
import tensor_cpt as tc
import scipy.io as scio
os.sys.path.append('./ppca-master/src/')
import pca
from PPCA import *
from sklearn.model_selection import train_test_split

#Truncated SVD
def T_SVD(Atensor,p):
    SD = dtensor(Atensor.copy())
    N = len(np.shape(SD))
    U_list,r_list = [],[]
    SG = []
    for i in range(N):
        B = SD.unfold(i)
        U,sigma,VT = scipy.linalg.svd(B,0)
        row_s = len(sigma)
        mat_sig = np.zeros((row_s,row_s))
        for j in range(row_s):
            mat_sig[j,j] = sigma[j]
            if sum(sigma[:j])/sum(sigma) > p:
                SG.append(sigma[j])
                break

        U_list.append(U[:,:j])
        r_list.append(j)
    return SG,j,U_list,r_list

def traffic_info(data):
    #data, miss_data1, W1, ori_W1 = handle_info
    #data = miss_data1  # change
    ds = data.shape
    var_mat = np.zeros((ds[0], ds[1]))
    mean_mat = np.zeros((ds[0], ds[1]))
    if len(ds) == 3:
        for r in range(ds[0]):
            for d in range(ds[1]):
                var_mat[r, d] = np.var(data[r, d, :])
                mean_mat[r, d] = np.mean(data[r, d, :])
    else:
        for r in range(ds[0]):
            std = np.std(data[r])
            mean = np.mean(data[r])
            for d in range(ds[1]):
                var_mat[r, d] = (data[r, d] - mean) / std
                mean_mat[r, d] = data[r, d]
    return var_mat,mean_mat

#tucker�ֽ����䷽��
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
    #ttm:����˷�
    ttm_data = core1.ttm(U1[0],0).ttm(U1[1],1).ttm(U1[2],2)
    print(np.linalg.norm(ttm_data-sparse_data))
    #print(np.mean(ttm_data))
    est_data = W*est_data+(W==False)*ttm_data
    time_e = time.time()
    print('-'*8+'tucker'+'-'*8)
    print('exec_time:'+str(time_e-time_s)+'s')
    return est_data

def multi_tucker(sparse_data,rates,W):
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
        est_dict[rate] = tucker_cpt(sparse_data,rank_set,W)
    return est_dict

#STD���
def STD_cpt(sparse_data,W,threshold=1e-4, alpha=2e-10, lm=0.01, p=0.7):
    ds = sparse_data.shape
    X_ori = sparse_data.copy()
    U_list,r_list = T_SVD(X_ori,p)[-2:]
    print(r_list)
    #r_list = [70,16,34]
    core, U_list = tucker.hooi(dtensor(X_ori), r_list, init='nvecs')
    [A,B,C] = U_list
    #core = dtensor(X_ori).ttm(A.T, 0).ttm(B.T, 1).ttm(C.T, 2)
    X = core.ttm(A, 0).ttm(B, 1).ttm(C, 2)
    #print(np.linalg.norm(X-X_ori))
    #return
    Upre_list = U_list
    F_diff = sys.maxsize
    iter = 0
    while F_diff > threshold and iter < 500:
        X_pre = X.copy()
        #print('Xpre_norm',np.linalg.norm(X_pre))
        # Upre_list = []
        # for u in U_list:
        #     Upre_list.append(u.copy())
        core_pre = core.copy()
        E = W*(X_ori-core_pre.ttm(Upre_list[0], 0).ttm(Upre_list[1], 1).ttm(Upre_list[2], 2))
        for i in range(X.ndim):
            mul1 = (W*E).unfold(i)
            if i == 0:
                mul2 = np.kron(Upre_list[2],Upre_list[1])
            elif i == 1:
                mul2 = np.kron(Upre_list[2],Upre_list[0])
            else:
                mul2 = np.kron(Upre_list[1],Upre_list[0])
            mul3 = core_pre.unfold(i).T
            Upre_list[i] = (1-alpha*lm)*Upre_list[i]+alpha*np.dot(np.dot(mul1,mul2),mul3)
            #print(np.dot(mul1,mul2))
        Temp = E.ttm(Upre_list[0].T, 0).ttm(Upre_list[1].T, 1).ttm(Upre_list[2].T, 2)
        core = (1-alpha*lm)*core_pre+alpha*Temp
        X = core.ttm(Upre_list[0],0).ttm(Upre_list[1],1).ttm(Upre_list[2],2)
        F_diff = np.linalg.norm(X-X_pre)
       #break
        iter += 1
    return X

#CP�ֽ����䷽��
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

#LRTC����䷽��
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
            para_fi = gama[n]/(alpha[n]+beta[n])
            U,sigma,VT = np.linalg.svd(M_temp,full_matrices=0)
            row_s = len(sigma)
            mat_sig = np.zeros((row_s,row_s))
            max_rank = 0
            for ii in range(row_s):
                mat_sig[ii,ii] = max(sigma[ii]-para_fi,0)
            M[n] = np.dot(np.dot(U[:,:row_s],mat_sig),VT[:row_s,:])
            M_fold[n] = M[n].fold()
        X = np.sum([alpha[i]*M_fold[i] for i in range(N)],axis=0)/sum(alpha)
        Y_temp = np.sum([beta[i]*M_fold[i] for i in range(N)],axis=0)/sum(beta)
        Y[W==False] = Y_temp[W==False]
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
            M[i] = (np.dot(np.dot(U,mat_sig),VT[:row_s,:])).fold()
        X_temp = np.sum([beta[j]*M[j] for j in range(N)],axis=0)/sum(beta)
        X[W==False] = X_temp[W==False]
        X_diffnorm = np.sum((X-X_pre)**2)
        if X_diffnorm < conv_thre:
            break
    time_e = time.time()
    print('-'*8+'silrtc'+'-'*8)
    print('exec_time:'+str(time_e-time_s)+'s')
    return X

def halrtc_cpt(sparse_data,lou,conv_thre,K,W,alpha=[0,0,1]):
    #ori_speeddata = scio.loadmat('../GZ_data/speed_tensor.mat')['tensor']
    #ori_speeddata, ori_W = tc.deal_orimiss(ori_speeddata, False)
    time_s = time.time()
    X = sparse_data.copy()
    Y = {}
    N = len(np.shape(X))
    W1 = (W==False)
    M = {}
    T_temp = X.copy()
    #alpha = [1/4,1/4,1/4,1/4] if N==4 else [0,0,1]
    for _ in range(N):
        M[_] = dtensor(np.zeros(np.shape(X)))
        Y[_] = dtensor(np.zeros(np.shape(X)))
    for iter in range(K):
        X_pre = X.copy()
        T_temp_pre = T_temp.copy()
        for i in range(N):
            SD = dtensor(X_pre)
            Matrix = SD.unfold(i)+1/lou*(Y[i].unfold(i))

            U,sigma,VT = np.linalg.svd(Matrix,0)
            row_s = len(sigma)
            mat_sig = np.zeros((row_s,row_s))
            for ii in range(row_s):
                mat_sig[ii,ii] = max(sigma[ii]-alpha[i]/lou,0)
            M[i] = (np.dot(np.dot(U,mat_sig),VT[:row_s,:])).fold()

        T_temp = (np.sum([M[j]-1/lou*Y[j] for j in range(N)],axis=0))/N
        X[W1] = T_temp[W1]
        X_Fnorm = np.sum((X-X_pre)**2)
        if X_Fnorm < conv_thre:
            break
        for i in range(N):
            Y[i] -= lou*(M[i]-X)
    time_e = time.time()
    #print('-'*8+'halrtc'+'-'*8)
    #print('exec_time:'+str(time_e-time_s)+'s')
    print('iter:',iter)
    return X

def PPCA_cpt(sparse_data,p=0.7):
    time_s = time.time()
    ds = sparse_data.shape
    est_PPCA = np.zeros_like(sparse_data)
    SD = sparse_data.copy()
    for i in range(ds[0]):
        data = SD[i]
        mult_pca_components = T_SVD(data, p)[-1][0]
        ppca = pca.ppca.PPCA(q=mult_pca_components)
        ppca.fit(data)
        est_PPCA[i] = ppca.transform_infers()
    time_e = time.time()
    print('-' * 8 + 'PPCA' + '-' * 8)
    print('exec_time:' + str(time_e - time_s) + 's')
    return est_PPCA

def BPCA_cpt(sparse_data,p=0.7):
    time_s = time.time()
    ds = sparse_data.shape
    est_BPCA = np.zeros_like(sparse_data)
    SD = sparse_data.copy()
    for i in range(ds[0]):
        data = SD[i]
        mult_pca_components = T_SVD(data, p)[-1][0]
        bppca = pca.bppca.BPPCA(data,q=mult_pca_components)
        bppca.fit()
        est_BPCA[i] = bppca.transform_infers()
    time_e = time.time()
    print('-' * 8 + 'BPCA' + '-' * 8)
    print('exec_time:' + str(time_e - time_s) + 's')
    return est_BPCA

def cluster_ha(labels,sparse_data,W,cluster_num,halrtc_para, alpha):
    sd = sparse_data.copy()
    Clr_mat = {i:[] for i in range(cluster_num)}
    for i in range(len(labels)):
        Clr_mat[labels[i]].append(i)
    [lou,K,conv_thre] = halrtc_para
    WT = W.copy()
    est_data = np.zeros_like(sd)
    for j in range(cluster_num):
        m_data = sd[labels == j]
        Wm = WT[labels == j]
        temp_data = halrtc_cpt(m_data,lou,conv_thre,K,Wm,alpha)
        est_data[Clr_mat[j]] = temp_data
    return est_data

def Kmeans_ha(sparse_data,W, K_n, K, conv_thre, p, alpha=[0,0,1]):
    SD = sparse_data.copy()
    time0 = time.time()
    lou = 1/T_SVD(SD,p)[0][0]
    #alpha = [0,0,1]
    halrtc_para = [lou,K,conv_thre]
    var_mat, mean_mat = traffic_info(SD)
    clf = KMeans(n_clusters=K_n)
    S = clf.fit(var_mat)
    L = S.labels_
    est_Kmeans = cluster_ha(L, SD, W, K_n, halrtc_para, alpha)
    time1 = time.time()
    print('-' * 8 + 'Kmeans++_ha' + '-' * 8)
    print('exec_time:' + str(time1 - time0) + 's')
    return est_Kmeans
