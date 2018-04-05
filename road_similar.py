#encoding=utf-8
import math
import sys
import scipy.io as scio
import random
import numpy as np

def cos_dis(mat1,mat2):
    if mat1.shape != mat2.shape:
        print('sizes are diffrent')
        return -1
    if np.isnan(mat1.any()) or np.isnan(mat2.any()):
        print('nan data')
        return -1
    ds = mat1.shape
    sim = 0
    #print(np.isnan(mat2).sum())
    for d in range(ds[0]):
        norm_product = np.linalg.norm(mat1[d])*np.linalg.norm(mat2[d])
        if np.isnan(norm_product) or norm_product == 0:
            print('norm error')
            return -1
        dis_pre = np.sum(mat1[d]*mat2[d])/norm_product
        if np.isnan(dis_pre):
            print('product error')
            print(mat2.sum())
            return -1
        sim += dis_pre
    sim = sim/ds[0]
    #按整个矩阵计算余弦相似,返回倒数表示值越小越相似
    #sim = np.sum(mat1*mat2)/(np.linalg.norm(mat1)*np.linalg.norm(mat2))
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

def road_Kmeans(sparse_data,ori_W,K_n,W):
    ds = sparse_data.shape
    Wn = W & ori_W
    W_cn = np.ones((ds[1],ds[2]))
    W_cn = np.tile(np.array([True]),(ds[1],ds[2]))
    #得出公共数据的位置
    for i in range(ds[0]):
        W_cn = W_cn & Wn[i]
    print(W_cn.sum())
    Init = random.sample(list(range(ds[0])),K_n)
    #W_cn = W_cn.astype(np.int)
    print((Init))
    Clr_center = np.zeros((K_n,ds[1],ds[2]))
    for i in range(K_n):
        Clr_center[i] = sparse_data[Init[i]]
    print(np.sum(Clr_center,axis=(1,2)))
    clr_go = True
    clr_assign = np.zeros((ds[0],2))
    while clr_go:
        clr_go = False
        for i in range(ds[0]):
            min_dis = 10000
            minIndex = -1
            for j in range(K_n):
                M1,M2 = sparse_data[i]*W_cn,Clr_center[j]*W_cn
                if np.isnan(np.linalg.norm(M1+M2)):
                    print(i,j)
                distance = cos_dis(M1,M2)
                if math.isnan(distance) or distance == 0:
                    print('nan:',i,Init[j])
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
    print(Clr_center.shape)
    return Clr_center,clr_assign


