#encoding=utf-8
import math
import time
import sys
import scipy.io as scio
import random
from random import shuffle
import numpy as np
from sklearn.cluster import spectral_clustering
from sklearn.cluster import KMeans
from sklearn import metrics

def Eu_dis(mat1,mat2):
    return np.linalg.norm(mat1-mat2)

def Mahalanobis(vec1,vec2):
    npvec1,npvec2 = np.array(vec1),np.array(vec2)
    npvec = np.array([npvec1,npvec2])
    sub = npvec.T[0]-npvec.T[1]
    try:
        inv_sub = np.linalg.inv(np.cov(npvec1,npvec2))
        res = math.sqrt(np.dot(inv_sub,sub).dot(sub.T))
    except:
        res = 0.1
    return res

def Ma_dis(mat1,mat2):
    ds = mat1.shape[0]
    return sum([Mahalanobis(mat1[i],mat2[i]) for i in range(ds)])

def cos_dis(mat1,mat2):
    if mat1.shape != mat2.shape:
        print('sizes are diffrent')
        return -1
    if np.isnan(mat1.any()) or np.isnan(mat2.any()):
        print('nan data')
        return -1
    ds = mat1.shape
    sim = 0
    #���������������������,���ص�����ʾֵԽСԽ����
    mat1 = mat1-np.mean(mat1)
    mat2 = mat2-np.mean(mat2)
    sim = np.sum(mat1*mat2)/(np.linalg.norm(mat1)*np.linalg.norm(mat2))
    return sim

def road_sim(M1,M2,sim_thre,sim_func='cos'):
    #��������·�ε�������
    if len(M1) != len(M2):
        print('different length')
        return -1
    N = len(M1)
    count = 0
    for i in range(N):
        if sim_func == 'cos':
            S = cos_dis(M1[i],M2[i])
        elif sim_func == 'pearson':
            S = pearson_dis(M1[i],M2[i])
        elif sim_func == "Eu":
            S = Eu_dis(M1[i],M2[i])
        #print('sim:',S)
        if S > sim_thre:
            count += 1
    return count/N

def pearson_dis(mat1,mat2):
    if mat1.shape != mat2.shape:
        print('sizes are diffrent')
        return -1
    M1 = mat1-np.mean(mat1)
    M2 = mat2-np.mean(mat2)
    norm1 = np.linalg.norm(M1)
    norm2 = np.linalg.norm(M2)
    sim = np.mean(M1*M2)/(norm1*norm2)
    return sim

def cmn_pos(sparse_data,ori_W,W,axis=0):
    ds = list(sparse_data.shape)
    ds[0],ds[axis] = ds[axis],ds[0] 
    Wn = W&ori_W
    Wn = Wn.swapaxes(0,axis)
    W_cn = np.ones((ds[1],ds[2]))
    W_cn = np.tile(np.array([True]),(ds[1],ds[2]))
    for i in range(ds[0]):
        W_cn = W_cn&Wn[i]
    return W_cn

def road_Kmeans(sparse_data,ori_W,K_n,W,axis=0,method='Eu'):
    SD = sparse_data.copy()
    SD = SD.swapaxes(0,axis)
    ds = SD.shape
    W_cn = cmn_pos(sparse_data,ori_W,ori_W,axis)
    #W_cn = W_cn.swapaxes(0,axis)
    Init = random.sample(list(range(ds[0])),K_n)
    Clr_center = np.zeros((K_n,ds[1],ds[2]))
    for i in range(K_n):
        Clr_center[i] = SD[Init[i]]
    clr_go = True
    clr_assign = np.zeros((ds[0],2))
    count = 0
    while clr_go and count < 40:
        count += 1
        clr_go = False
        for i in range(ds[0]):
            min_dis = sys.maxsize
            minIndex = -1
            for j in range(K_n):
                M1,M2 = SD[i],Clr_center[j]
                if np.isnan(M2).all():
                    print(i,j)
                    Clr_center = np.delete(Clr_center,j,0)
                    K_n -= 1
                    break
                if method == 'Eu':
                    distance = dis_method[method](M1,M2)/1000
                else:
                    distance = 1-dis_method[method](M1,M2)
                #distance = 1-cos_dis(M1,M2)
                #distance = 1-pearson_dis(M1,M2)
                #distance = 1/Ma_dis(M1,M2)
                #distance = 1-road_sim(M1,M2,0.75,'pearson')
                if distance < min_dis:
                    min_dis = distance
                    minIndex = j
            #�ҵ����������
            if clr_assign[i,0] != minIndex:
                clr_assign[i,:] = minIndex,min_dis
                clr_go = True
        for j in range(K_n):
            clr_points = SD[(clr_assign[:,0]==j)]
            Clr_center[j] = np.mean(clr_points,axis=0)
    #print(clr_assign)
    return clr_assign.T[0],K_n

def getWbyKNN(data_,k,w_dis,method='Eu',axis=0):
    data = data_.copy()
    data = data.swapaxes(0,axis)
    points_num = len(data)
    dis_matrix = np.zeros((points_num,points_num))
    W = np.zeros_like(dis_matrix)
    for i in range(points_num):
        for j in range(i+1,points_num):
            dist = dis_method[method](data[i],data[j])
            if method in ('Eu','multi'):
                dist = 1/dist
            dis_matrix[i][j] = dis_matrix[j][i] = dist
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

def SC_1(sparse_data,KNN_k,cluster_num,w_dis=[0.01,0.1,0.2],method='Eu',axis=0):
    WM = getWbyKNN(sparse_data,KNN_k,w_dis,method,axis)
    DM = getD(WM)
    LM = DM-WM
    eigval,eigvec = getEigVec(LM,cluster_num)
    clf = KMeans(n_clusters=cluster_num)
    s = clf.fit(eigvec)
    C = s.labels_
    return C

def U2C(X,U,m):
    dX,dU = X.shape,U.shape
    S = np.zeros((dU[0],dX[1],dX[2]))
    for i in range(dU[0]):
        for j in range(dU[1]):
            S[i] += (U[i,j]**m)*X[j,:,:]
    T = np.sum(U**m,axis=1)
    C = []
    for j in range(dU[0]):
        c = S[j]/T[j]
        if np.isnan(c).all():
            break
        C.append(c)
    return np.array(C)

def XC_dis(X,C,m,method):
    D = np.zeros((len(C),len(X)))
    for i in range(len(C)):
        for j in range(len(X)):
            if method in ('Eu','multi'):
                D[i,j] = dis_method[method](X[j],C[i])+1
            else:
                D[i,j] = 2-dis_method[method](X[j],C[i])
    return D

def C2U(X,C,m,method):
    D = XC_dis(X,C,m,method)
    U = np.zeros_like(D)
    for i in range(len(C)):
        for j in range(len(X)):
            U[i,j] = D[i,j]**(-2/(m-1))/np.sum(D[:,j]**(-2/(m-1)))
    return U

def target_func(U,X,C,m,method):
    D = XC_dis(X,C,m,method) 
    #print(np.isinf(U).all(),print(np.isinf(D).all()))
    J = np.sum((U**m)*(D**2))
    return J

def fcm(X,cluster_num,m=1.5,method='pearson'):
    C_init = X.copy()
    NUM = list(range(len(X)))
    shuffle(NUM)
    C = C_init[NUM[:cluster_num]]
    #C = INIT_C(cluster_num,X)
    J,J_de,J_pre = 0,1,-10
    cnt = 0
    while(abs(J-J_pre)>J_de and cnt<200):
        J_pre = J
        U = C2U(X,C,m,method)
        J = target_func(U,X,C,m,method)
        C = U2C(X,U,m)
        cnt += 1
    return U,C

def multi_dis(M1,M2,w_dis=[0.01,0.1,0.2]):
    days, periods = M1.shape
    X = [M1,M2]
    para_matrix = np.zeros((2, days, 3))
    for r in range(2):
        para_matrix[r, :, 0] = np.linalg.svd(X[r], 0)[1]
        para_matrix[r, :, 1] = [np.var(X[r][d]) for d in range(days)]
        para_matrix[r, :, 2] = [np.mean(X[r][d]) for d in range(days)]
    return sum([Eu_dis(para_matrix[0,:,i],para_matrix[1,:,i])*w_dis[i] for i in range(3)])

def multi_sim(X,w_dis=[0.01,0.1,0.2]):
    roads,days,periods = X.shape
    para_matrix = np.zeros((roads,days,3))
    dis_matrix = np.zeros((roads,roads))
    for r in range(roads):
        para_matrix[r,:,0] = np.linalg.svd(X[r],0)[1]
        para_matrix[r,:,1] = [np.var(X[r,d]) for d in range(days)]
        para_matrix[r,:,2] = [np.mean(X[r,d]) for d in range(days)]
    for r1 in range(roads):
        for r2 in range(roads):
            dis_matrix[r1,r2] = sum([Eu_dis(para_matrix[r1,:,i],para_matrix[r2,:,i])*w_dis[i] for i in range(3)])
    return dis_matrix

dis_method = {'Eu':Eu_dis,'cos':cos_dis,'pearson':pearson_dis,'multi':multi_dis}
