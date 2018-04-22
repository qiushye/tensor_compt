#encoding=utf-8
import math
import sys
import scipy.io as scio
import random
import numpy as np
from sklearn.cluster import spectral_clustering
from sklearn.cluster import KMeans
from sklearn import metrics

def Eu_dis(mat1,mat2):
    return np.linalg.norm(mat1-mat2)

def cos_dis(mat1,mat2):
    if mat1.shape != mat2.shape:
        print('sizes are diffrent')
        return -1
    if np.isnan(mat1.any()) or np.isnan(mat2.any()):
        print('nan data')
        return -1
    ds = mat1.shape
    sim = 0
    #按整个矩阵计算余弦相似,返回倒数表示值越小越相似
    sim = np.sum(mat1*mat2)/(np.linalg.norm(mat1)*np.linalg.norm(mat2))
    return 1/(sim+1)

def pearson_dis(mat1,mat2):
    if mat1.shape != mat2.shape:
        print('sizes are diffrent')
        return -1
    M1 = mat1-np.mean(mat1)
    M2 = mat2-np.mean(mat2)
    norm1 = np.linalg.norm(M1)
    norm2 = np.linalg.norm(M2)
    sim = np.sum(M1*M2)/(norm1*norm2)
    return 1/(sim+0.1)

def cmn_pos(sparse_data,ori_W,W):
    ds = sparse_data.shape
    Wn = W&ori_W
    W_cn = np.ones((ds[1],ds[2]))
    W_cn = np.tile(np.array([True]),(ds[1],ds[2]))
    for i in range(ds[0]):
        W_cn = W_cn&Wn[i]
    return W_cn

def road_Kmeans(sparse_data,ori_W,K_n,W):
    ds = sparse_data.shape
    W_cn = cmn_pos(sparse_data,ori_W,ori_W)
    Init = random.sample(list(range(ds[0])),K_n)
    #W_cn = W_cn.astype(np.int)
    print((Init))
    Clr_center = np.zeros((K_n,ds[1],ds[2]))
    for i in range(K_n):
        Clr_center[i] = sparse_data[Init[i]]
    print(np.sum(Clr_center,axis=(1,2)))
    clr_go = True
    clr_assign = np.zeros((ds[0],2))
    count = 0
    while clr_go:
        count += 1
        clr_go = False
        for i in range(ds[0]):
            min_dis = sys.maxsize
            minIndex = -1
            for j in range(K_n):
                M1,M2 = sparse_data[i]*W_cn,Clr_center[j]*W_cn
                if np.isnan(M1+M2).all():
                    print(i,j)
                #distance = Eu_dis(M1,M2)/1000
                distance = cos_dis(M1,M2)
                if distance < min_dis:
                    min_dis = distance
                    minIndex = j
            #找到最近的中心
            if clr_assign[i,0] != minIndex:
                clr_assign[i,:] = minIndex,min_dis**2
                clr_go = True
        for j in range(K_n):
            clr_points = sparse_data[(clr_assign[:,0]==j)]
            Clr_center[j] = np.mean(clr_points,axis=0)
        #break
    print(clr_assign)
    print('count:',count)
    print(Clr_center.shape)
    return Clr_center,clr_assign

def getWbyKNN(data,k):
    points_num = len(data)
    dis_matrix = np.zeros((points_num,points_num))
    W = np.zeros_like(dis_matrix)
    for i in range(points_num):
        for j in range(i+1,points_num):
            dis_matrix[i][j] = dis_matrix[j][i] = pearson_dis(data[i],data[j])
    for idx,each in enumerate(dis_matrix):
        index_array = np.argsort(each)
        W[idx][index_array[1:k+1]] = 1
    tmp_W = np.transpose(W)
    W = (tmp_W+W)/2
    return W

def getD(W):
    points_num = len(W)
    D = np.diag(np.zeros(points_num))
    for i in range(points_num):
        D[i][i] = sum(W[i])
    return D

def getEigVec(L,cluster_num):
    eigval,eigvec = np.linalg.eig(L)
    dim = len(eigval)
    dictEigval = dict(zip(eigval,range(0,dim)))
    kEig = np.sort(eigval)[0:cluster_num]
    ix = [dictEigval[k] for k in kEig]
    return eigval[ix],eigvec[:,ix]

def SC(sparse_data,ori_W,W,cluster_num):
    W_n = cmn_pos(sparse_data,ori_W,ori_W)
    Data_n = np.zeros((W.shape[0],W_n.sum()))
    print(Data_n.shape,W_n.sum())
    for i in range(W.shape[0]):
        p = 0
        for j in range(W.shape[1]):
            for k in range(W.shape[2]):
                if W_n[j,k]:
                    Data_n[i,p] = sparse_data[i,j,k]
                    p += 1
        print(p)
        break
    M = -1*metrics.pairwise.pairwise_distances(Data_n)
    M += -1*M.min()
    labels = spectral_clustering(M,n_clusters=cluster_num)
    return labels

def SC_1(sparse_data,KNN_k,cluster_num):
    WM = getWbyKNN(sparse_data,KNN_k)
    DM = getD(WM)
    LM = DM-WM
    eigval,eigvec = getEigVec(LM,cluster_num)
    clf = KMeans(n_clusters=cluster_num)
    s = clf.fit(eigvec)
    C = s.labels_
    return C
