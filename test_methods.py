#encoding=utf-8
#import tensor_cpt
from tensor_cpt import deal_orimiss,pre_impute,rmse_mape_rse,gene_rand_sparse,get_sparsedata,gene_sparse
from compt_methods import halrtc_cpt,cp_cpt,silrtc_cpt,T_SVD,traffic_info, cluster_ha
from road_similar import cos_dis,Eu_dis,pearson_dis,fcm,road_Kmeans,SC_1
from sklearn.cluster import AgglomerativeClustering,KMeans
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage,dendrogram,cophenet,fcluster
from sktensor import dtensor
import numpy as np
import os
import scipy.io as scio
import sys
import time
import matplotlib.pyplot as plt
import random

def handle_data(data_,mr=0.2,miss_type = 'rand'):
    data = data_.copy()
    data, ori_W1 = deal_orimiss(data)
    ds = data.shape
    #mr = 0.2
    miss_path1 = data_dir + '_'.join(['miss',miss_type,str(round(mr,1))]+[str(ch) for ch in ds]) + '.mat'
    if not os.path.exists(miss_path1):
        gene_sparse[miss_type](data, mr, miss_path1)
    miss_data1, W_, tr = get_sparsedata(miss_path1)
    W1 = miss_data1 > 0
    miss_data1 = pre_impute(miss_data1, W1)
    return data,miss_data1,W1,ori_W1

def week_sepe(handle_train,handle_test):
    data, miss_data1, W1, ori_W1 = handle_train
    halrtc_para = [lou, K, conv_thre, fb] = [3e-3, 100, 1e-5]
    data_test, miss_data_test, Wtest, ori_Wtest = handle_test
    ds_test = data_test.shape
    est_ori = halrtc_cpt(miss_data_test, lou, conv_thre, K, Wtest, fb)
    weekday2, weekend2 = [], []
    for i in range(ds_test[1]):
        if i % 7 < 5:
            weekday2.append(i)
        else:
            weekend2.append(i)
    est1 = halrtc_cpt(miss_data_test[:,weekday2,:], lou, conv_thre, K, Wtest[:,weekday2,:], fb)
    eva1 = rmse_mape_rse(est1, data_test[:,weekday2,:], (Wtest | (ori_Wtest == False))[:,weekday2,:])
    est2 = halrtc_cpt(miss_data_test[:, weekend2, :], lou, conv_thre, K, Wtest[:, weekend2, :], fb)
    eva2 = rmse_mape_rse(est2, data_test[:, weekend2, :], (Wtest | (ori_Wtest == False))[:, weekend2, :])
    print('weekday',eva1)
    print('ori_weekday',rmse_mape_rse(est_ori[:,weekday2,:], data_test[:,weekday2,:], (Wtest | (ori_Wtest == False))[:,weekday2,:]))
    print('weekend',eva2)
    print('ori_weekend', rmse_mape_rse(est_ori[:, weekend2, :], data_test[:, weekend2, :], (Wtest | (ori_Wtest == False))[:, weekend2, :]))
    return


def test_simM(data_,r,dis_matrix,k):
    r_dis = dis_matrix[r]
    dis_dict = {r_dis[i]:i for i in range(len(r_dis))}
    sim_road = [dis_dict[j] for j in sorted(dis_dict.keys())[:k+1]]
    data, miss_data1, W1, ori_W1 = handle_data(data_)
    lou, K, conv_thre, fb = 3e-3, 100, 1e-5, 0.85
    est_ori = halrtc_cpt(miss_data1, lou, conv_thre, K, W1, fb)
    eva_ori = rmse_mape_rse(est_ori[r], data[r], (W1 | (ori_W1 == False))[r])
    est_sim = halrtc_cpt(miss_data1[sim_road], lou, conv_thre, K, W1[sim_road], fb)
    eva_sim = rmse_mape_rse(est_sim[0], data[r], (W1 | (ori_W1 == False))[r])
    print(eva_ori)
    print(eva_sim)
    return


def var_sepe(data_):
    ds = data_.shape
    data, miss_data1, W1, ori_W1 = handle_data(data_)
    lou, K, conv_thre, fb = 3e-3, 100, 1e-5, 0.85
    est_ori = halrtc_cpt(miss_data1, lou, conv_thre, K, W1, fb)
    var_list = []
    mean_list,svd_list = [],[]
    for r in range(ds[0]):
        var_list.append(np.var(data[r]))
        mean_list.append(np.mean(data[r]))
    var_dict = {var_list[i]: i for i in range(ds[0])}
    var_assign = [[],[],[]]
    for var in var_dict:
        if var < 50:
            var_assign[0].append(var_dict[var])
        elif var < 150:
            var_assign[1].append(var_dict[var])
        else:
            var_assign[2].append(var_dict[var])
    est_var = np.zeros_like(data)
    for va in var_assign:
        est_va = halrtc_cpt(miss_data1[va], lou, conv_thre, K, W1[va], fb)
        est_var[va] = est_va
    eva_var = rmse_mape_rse(est_var, data, (W1 | (ori_W1 == False)))
    print('sepe',eva_var)
    est_ori = halrtc_cpt(miss_data1, lou, conv_thre, K, W1, fb)
    eva_ori = rmse_mape_rse(est_ori, data, (W1 | (ori_W1 == False)))
    print('ori',eva_ori)
    return

def var_cluster(handle_train):
    data, miss_data1, W1, ori_W1 = handle_train
    ds = data.shape
    var_mat = np.zeros(ds[:2])
    for r in range(ds[0]):
        for d in range(ds[1]):
            var_mat[r,d] = np.var(data[r,d,:])
    cn = 2
    clf = KMeans(n_clusters=cn)
    s = clf.fit(var_mat)
    C = s.labels_
    return C

def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
        annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)
    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5), textcoords='offset points', va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
        return ddata

def test_labels(cn,labels,handle_info):
    halrtc_para = [lou, K, conv_thre, fb] = [3e-3, 100, 1e-5]
    data_test, miss_data_test, Wtest, ori_Wtest = handle_info
    ds_test = data_test.shape
    est_ori = halrtc_cpt(miss_data_test, lou, conv_thre, K, Wtest, fb)
    for i in range(cn):
        print('label-'+str(i),(labels==i).sum())
        temp_data = halrtc_cpt(miss_data_test[labels==i], lou, conv_thre, K, Wtest[labels==i], fb)
        #est_c = cluster_ha(labels, miss_data_test, Wtest, cn, halrtc_para, axis=0)
        eva_c = rmse_mape_rse(temp_data, data_test[labels==i], (Wtest | (ori_Wtest == False))[labels==i])
        print('clu', eva_c)
        eva_test = rmse_mape_rse(est_ori[labels==i], data_test[labels==i], (Wtest | (ori_Wtest == False))[labels==i])
        print('ori',eva_test)
    return

def test_week(data_,test_data_,sepe_day):
    weekday1, weekend1 = [], []
    weekday2, weekend2 = [], []
    for i in range(sepe_day):
        if i % 7 < 5:
            weekday1.append(i)
        else:
            weekend1.append(i)
    for i in range(sepe_day, 61):
        if i % 7 < 5:
            weekday2.append(i - sepe_day)
        else:
            weekend2.append(i - sepe_day)
    #var_cluster(data_[:,weekend1,:], test_data_[:,weekend2,:])
    return


def AHC(handle_info,choise,var_mat,mean_mat,k=2,max_d=100):
    #var_mat,mean_mat = traffic_info(handle_info)
    # 建立集群关系数组
    if choise == 'var':
        Z = linkage(var_mat,'complete')
        c, coph_dists = cophenet(Z, pdist(var_mat))
    elif choise == 'mean':
        Z = linkage(mean_mat, 'complete')
        c, coph_dists = cophenet(Z, pdist(mean_mat))
    print('同表象相关系数',c)
    #print(Z[:10])
    plt.figure(figsize=(50, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    #max_d = 50
    #k = 2
    fancy_dendrogram(Z, truncate_mode='lastp', p=12, leaf_rotation=90., leaf_font_size=12., show_contracted=True,\
                     annotate_above=10,max_d=max_d)
    #clusters = fcluster(Z, max_d, criterion='distance')
    clusters = fcluster(Z, k, criterion='maxclust')
    print(clusters)
    plt.show()
    plt.savefig('ahc_tree.png')
    plt.close()
    return clusters-1,c

def test_cluster(handle_train,handle_test,choise = 'var'):
    p_tr = 0.9
    halrtc_para = [lou, K, conv_thre, fb] = [1.3e-3, 100, 1e-5]
    data_test, miss_data_test, Wtest, ori_Wtest = handle_test
    rW = Wtest | (ori_Wtest == False)
    time_a = time.time()
    #lou = 1/(3*T_SVD(miss_data_test,p_tr)[0])
    est_ori = halrtc_cpt(miss_data_test, lou, conv_thre, K, Wtest, fb)
    eva_ori = rmse_mape_rse(est_ori, data_test, (Wtest | (ori_Wtest == False)))
    K_n = 2
    max_d = 100
    time_s = time.time()
    ori_time = time_s - time_a
    print('ori', eva_ori)
    print('ori_time', str(ori_time) + 's')
    ts = data_test.shape
    L = np.zeros(ts[0])
    Road = np.array(range(ts[0]))
    max_c = 0
    est_clu = np.zeros_like(data_test)
    #handle_train = handle_test #change
    var_mat, mean_mat = traffic_info(handle_train)
    iter = 0
    while True:
        HD_test = []
        for temp in handle_test:
            HD_test.append(temp[L == max_c])
        var_mat = var_mat[L==max_c]
        mean_mat = mean_mat[L==max_c]
        L,c = AHC(handle_train,choise,var_mat,mean_mat,K_n,max_d)
        if c < 0.65 or iter > 3:
            break
        #test_labels(K_n,L,HD_test)
        HD_train = []
        max_c, max_n = -1, 0
        for i in range(K_n):
            if (L == i).sum() > max_n:
                max_c = i
                max_n = (L == i).sum()
        for i in range(K_n):
            if i != max_c:
                list_i = Road[L==i]
                lou = 1 / (3 * T_SVD(miss_data_test[list_i], p_tr)[0])
                est_clu[list_i] = halrtc_cpt(miss_data_test[list_i],
                                       lou, conv_thre, K, Wtest[list_i], fb)
                print('clu',rmse_mape_rse(est_clu[list_i],data_test[list_i],rW[list_i]))
                print('ori',rmse_mape_rse(est_ori[list_i],data_test[list_i],rW[list_i]))

        for temp in handle_train:
            HD_train.append(temp[L == max_c])
        handle_train = HD_train.copy()
        handle_test = HD_test.copy()
        Road = Road[L == max_c]
        iter += 1

        #print(Road)
        #if Road.size < 80:
        #    break
    #最后一部分聚类与否

    clf = KMeans(n_clusters=K_n)
    data, miss_data1, W1, ori_W1 = HD_train
    ds1 = data.shape
    mean_mat = np.zeros(ds1[:2])
    for r in range(ds1[0]):
        for d in range(ds1[1]):
            mean_mat[r, d] = np.mean(data[r, d, :])
            #mean_mat[r, d] = np.std(data[r,d,:])/np.mean(data[r,d,:])
    s = clf.fit(mean_mat)
    L = np.array(s.labels_)
    for i in range(K_n):
        list_i = Road[L == i]
        #lou = 1 / (3 * T_SVD(miss_data_test[list_i], p_tr)[0])
        est_clu[list_i] = halrtc_cpt(miss_data_test[list_i], lou, conv_thre, K, Wtest[list_i], fb)

    #lou = 1 / (3 * T_SVD(miss_data_test[Road], p_tr)[0])
    #est_clu[Road] = halrtc_cpt(miss_data_test[Road], lou, conv_thre, K, Wtest[Road], fb)

    eva_clu = rmse_mape_rse(est_clu, data_test, (Wtest | (ori_Wtest == False)))
    time_e = time.time()
    ori_time = time_s - time_a
    print('ori', eva_ori)
    print('ori_time', str(ori_time) + 's')
    print('clu', eva_clu)
    clu_time = time_e - time_s
    print('clu_time', str(clu_time) + 's')
    return eva_ori, ori_time, eva_clu, clu_time,est_clu

def test_week_cluster(handle_train,handle_test,choise='var'):
    #按一周七天分别聚类
    data, miss_data1, W1, ori_W1 = handle_train
    ds = data.shape
    train = {}
    halrtc_para = [lou, K, conv_thre, fb] = [3e-3, 100, 1e-5]
    data_test, miss_data_test, Wtest, ori_Wtest = handle_test
    est_ori = halrtc_cpt(miss_data_test, lou, conv_thre, K, Wtest, fb)
    ds_test = data_test.shape
    test,est_clu = {},{}
    est_week = np.zeros_like(data_test)
    K_n,max_d = 2,100
    for i in range(7):
        train[i],test[i] = [],[]
        days_train = []
        for k in range(ds[1]):
            if k%7 == i:
                days_train.append(k)
        for s in handle_train:
                train[i].append(s[:,days_train,:])

        days_test = []
        for k in range(ds_test[1]):
            if k%7 == i:
                days_test.append(k)
        for t in handle_test:
            test[i].append(t[:,days_test,:])
        L = np.zeros(ds[0])
        for j in range(ds[0]):
            if np.var(data[j,i])>50:
                L[j] = 1
        #L = SC_1(train[i][0], 6, K_n, method='pearson')
        est_clu[i] = cluster_ha(L, test[i][1], test[i][2], K_n, halrtc_para)
        #est_clu[i] = test_cluster(train[i], test[i])[-1]
        for p in range(est_clu[i].shape[1]):
            est_week[:,p*7+i,:] = est_clu[i][:,p,:]
        eva_week = rmse_mape_rse(est_clu[i], data_test[:,days_test,:], (Wtest | (ori_Wtest == False))[:,days_test,:])
        print('clu',i,eva_week)
        eva_i = rmse_mape_rse(est_ori[:,days_test,:], data_test[:,days_test,:], (Wtest | (ori_Wtest == False))[:,days_test,:])
        print('ori',i,eva_i)
    return

def AHC_mr(ori_speeddata,img_dir):
    sp_day = 7
    train_day = list([0]) + list(range(1, sp_day))
    train_data = ori_speeddata[:, train_day, :]
    handle_train = handle_data(train_data,0)
    test_data = ori_speeddata[:, sp_day:, :]
    res = np.zeros((2,8,4))
    mr_list = []
    eva_list = ['RMSE','MAPE','MAE','RT']
    mt = 'cont'
    for i in range(8):
        mr_test = 0.1*(i+1)
        mr_list.append(mr_test)
        handle_test = handle_data(test_data, mr_test,'rand')
        print('-----'+str(i)+'-----')
        eva_ori, ori_time, eva_clu, clu_time = compare_3d_4d(handle_train,handle_test)
        # eva_ori,ori_time,eva_clu,clu_time = test_cluster(handle_train, handle_test)
        res[0,i] = [eva_ori[0],eva_ori[1],eva_ori[3],ori_time]
        res[1,i] = [eva_clu[0],eva_clu[1],eva_clu[3],clu_time]
    for j in range(4):
        plt.plot(mr_list,res[0,:,j],'r')
        plt.plot(mr_list,res[1,:,j],'b')
        plt.title(eva_list[j])
        plt.savefig(img_dir+eva_list[j]+'_var_mr.png')
        plt.close()
    return

def corr_rela(handle_train,ori_speeddata):
    data, miss_data1, W1, ori_W1 = handle_train
    #data = dtensor(data).unfold(2)
    for i in range(data.ndim):
        data = dtensor(miss_data1).unfold(i)
        ds = data.shape
        S = 0
        for j in range(ds[0]):
            for k in range(j+1,ds[0]):
                C = pearson_dis(data[j,:],data[k,:])
                S += C
        S /= (ds[0]*(ds[0]-1)/2)
        print(str(i)+'_CC',S)
    return
    Y1 = []
    Y2 = []
    r = 60
    for i in range(ds[2]):
        Y1.append(np.var(data[r,:,i]))
        Y2.append(np.std(data[r,:,i])/np.mean(data[r,:,i]))
    plt.figure()
    #plt.plot(list(range(ds[1])),Y1,'r')
    plt.plot(list(range(ds[2])),Y2,'b')
    plt.savefig(img_dir+str(r)+'_corr_rela.png')
    plt.close()
    return

def clusters_train(handle_train,handle_test):
    K_n = 2
    halrtc_para = [lou,K,conv_thre,fb] = [1e-3, 100, 1e-5]
    data, miss_data1, W1, ori_W1 = handle_train
    data_test, miss_data_test, Wtest, ori_Wtest = handle_test
    est_ori = halrtc_cpt(miss_data_test, lou, conv_thre, K, Wtest, fb)
    time0 = time.time()
    var_mat,mean_mat = traffic_info(handle_test)
    clf = KMeans(n_clusters=K_n)
    S = clf.fit(var_mat)
    L = S.labels_
    print(L)
    '''
    L = np.zeros(209)
    var_list = []
    std_list = []
    var_mean = np.zeros((209,2))
    for i in range(209):
        #按方差均值划分
        #if np.mean(var_mat[i]) > 50:
        #t = np.std(data[i])/np.mean(data[i])
        t = np.mean(var_mat[i])
        var_list.append(t)
        var_mean[0] = np.var(data[i])
        var_mean[1] = np.mean(data[i])
        if t < 50:
            s = round(t**0.5/np.mean(data[i]),2)
            L[i] = 1
    return L
    '''
    #est_Kmeans[L==1] = est_ori[L==1]
    est_Kmeans = cluster_ha(L, miss_data_test, Wtest, K_n, halrtc_para)
    time1 = time.time()
    for i in range(K_n):
        print('ori_'+str(i),rmse_mape_rse(est_ori[L==i], data_test[L==i], (Wtest | (ori_Wtest == False))[L==i]))
        print('KM_' + str(i),
              rmse_mape_rse(est_Kmeans[L == i], data_test[L == i], (Wtest | (ori_Wtest == False))[L == i]))

    print('Kmeans:', rmse_mape_rse(est_Kmeans, data_test, Wtest | (ori_Wtest == False)))
    print('Kmeans_time:', str(time1 - time0) + 's')
    
    time0 = time.time()
    est_ori = halrtc_cpt(miss_data_test, lou, conv_thre, K, Wtest, fb)
    time1 = time.time()
    eva_ori = rmse_mape_rse(est_ori, data_test, (Wtest | (ori_Wtest == False)))
    print('ori:',eva_ori)
    print('ori_time:',str(time1-time0)+'s')
    '''
    time0 = time.time()
    L = SC_1(data,10,K_n,method='pearson')
    est_SC = cluster_ha(L,miss_data_test,Wtest,K_n,halrtc_para)
    time1 = time.time()
    print('SC:',rmse_mape_rse(est_SC,data_test,Wtest|(ori_Wtest==False)))
    print('SC_time:',str(time1-time0)+'s')
    time0 = time.time()
    L = road_Kmeans(data,ori_W1,K_n,W1,method='pearson')[0]
    est_Kmeans = cluster_ha(L, miss_data_test, Wtest, K_n,halrtc_para)
    time1 = time.time()
    print('Kmeans:', rmse_mape_rse(est_Kmeans, data_test, Wtest | (ori_Wtest == False)))
    print('Kmeans_time:',str(time1-time0)+'s')
    '''
    return

def plot_img(handle_train,ori_speeddata):
    data, miss_data1, W1, ori_W1 = handle_train
    ds = data.shape
    ds = ori_speeddata.shape
    var_list = []
    Monday_v = []
    for j in range(ds[1]):
        if j%7 != 2:
            continue
        Monday_list = []
        for k in range(ds[2]):
            Monday_list.append(ori_speeddata[0,j,k])
        plt.plot(list(range(ds[2])),Monday_list,color='blue',linestyle='-',label='j')
    plt.xlabel('Time interval (10 min)')
    plt.ylabel('Speed (km/h)')
    plt.savefig(img_dir+'all_Mondays.png')
    plt.close()
    # for i in range(ds[2]):
    #     Monday_v.append(data[0,2,i])
    # #plt.xlim(0,150)
    # plt.plot(list(range(ds[2])),Monday_v,'b')
    # plt.xlabel('Time interval (10 min)')
    # plt.ylabel('Speed (km/h)')
    # plt.savefig(img_dir+'Monday_1.png')
    # plt.close()
    return
    road = 1
    for j in range(ds[1]):
        v_list = []
        for k in range(ds[2]):
            v_list.append(data[road,j,k])
        plt.plot(list(range(ds[2])),v_list)
    plt.savefig(img_dir+str(road)+'_train_v.png')
    # for i in range(ds[0]):
    #     var_list.append(np.var(data[i]))
    # plt.plot(list(range(ds[0])),var_list)
    # plt.savefig(img_dir+'train_var.png')
    plt.close()
    return

def compare_3d_4d(handle_train,handle_test):
    data, miss_data1, W1, ori_W1 = handle_train
    halrtc_para = [lou,K,conv_thre,fb] = [1e-3, 100, 1e-5]
    data_test, miss_data_test, Wtest, ori_Wtest = handle_test
    rW = Wtest|(ori_Wtest==False)
    data_size = data_test.shape
    weeks = data_size[1] // 7
    '''
    Nori_data = np.zeros((data_size[0], weeks, 7, data_size[2]))
    Nmiss_data = np.zeros_like(Nori_data)
    N_W = np.zeros_like(Nmiss_data)
    Nori_W = np.zeros_like(Nori_data)
    Nr_W = np.zeros_like(Nori_W)
    for i in range(data_size[1]):
        if i >= weeks * 7:
            break
        Nori_data[:, i // 7, i % 7, :] = data_test[:, i, :]
        Nori_W[:, i // 7, i % 7, :] = ori_Wtest[:, i, :]
        Nmiss_data[:, i // 7, i % 7, :] = miss_data_test[:, i, :]
        N_W[:, i // 7, i % 7, :] = Wtest[:, i, :]
        Nr_W[:, i // 7, i % 7, :] = rW[:, i, :]
    print(Nmiss_data.shape)
    time0 = time.time()
    est_4d = np.zeros_like(Nmiss_data)
    est_4d[L==0] = halrtc_cpt(Nmiss_data[L==0], 1e-3, 1e-4, 100, N_W[L==0], 0)
    est_4d[L == 1] = halrtc_cpt(Nmiss_data[L == 1], 1e-3, 1e-4, 100, N_W[L==1], 0)
    time1 = time.time()
    print('4d_halrtc:', rmse_mape_rse(est_4d, Nori_data, Nr_W))
    print('4d_time', str(time1 - time0) + 's')
    '''
    print('loss',(miss_data_test<1).sum())
    time0 = time.time()
    est_ori = halrtc_cpt(miss_data_test, lou, conv_thre, K, Wtest, fb)
    time1 = time.time()
    eva_ori = rmse_mape_rse(est_ori, data_test, rW)
    print('ori:', eva_ori)
    time_ori = time1 - time0
    print('ori_time:', str(time_ori) + 's')
    return
    L = np.zeros(data_size[0])
    var_mat, mean_mat = traffic_info(handle_train)
    for i in range(data_size[0]):
        #按方差均值划分
        t = np.var(data[i])
        if t < 50:
            s = round(t**0.5/np.mean(data[i]),2)
            L[i] = 1
    #L = var_cluster(handle_train)


    time0 = time.time()
    est_3d = cluster_ha(L, miss_data_test, Wtest, 2, halrtc_para)
    time1 = time.time()
    eva_3d = rmse_mape_rse(est_3d, data_test,rW)
    print('3d:', eva_3d)
    time_3d = time1-time0
    print('3d_time:',str(time_3d)+'s')

    return eva_ori,time_ori,eva_3d,time_3d


if __name__ == '__main__':
    data_dir = './data/'
    img_dir = './img_test/'
    #mat_path = '/home/qiushye/2013_east/2013_east_speed.mat'
    # 广州数据
    #ori_speeddata = scio.loadmat('../GZ_data/speed_tensor.mat')['tensor']
    ori_speeddata = scio.loadmat('200_60_288.mat')['speed']
    print(ori_speeddata.shape)
    # ori_speeddata = ori_speeddata[:200,:60,:288]
    # scio.savemat('200_60_288.mat',{'speed':ori_speeddata})
    print((np.isnan(ori_speeddata).sum()))
    #ori_speeddata = ori_speeddata.swapaxes(0,2)
    #ori_speeddata = scio.loadmat('../qidong_data/qidong_speed.mat')['speed']
    #AHC_mr(ori_speeddata,img_dir)
    #sys.exit()
    sp_day = 7
    train_day = list([0])+list(range(1,7))
    train_data = ori_speeddata[:,train_day,:]
    handle_train = handle_data(train_data)
    test_data = ori_speeddata[:,:,:]
    print(test_data.shape)
    mr_test = 0.2
    handle_test = handle_data(test_data,mr_test,'cont')
    ods = ori_speeddata.shape
    #plot_img(handle_train,ori_speeddata)
    #corr_rela(handle_test,ori_speeddata)
    #disM = scio.loadmat('road_sim.mat')['sim']
    # train_sim(train_data)
    #week_sepe(handle_train,handle_test)
    #test_simM(train_data, 1, disM, 10)
    #var_rmse(train_data)
    #var_sepe(train_data)
    #test_lastgroup(handle_train,handle_test)
    #print(1/(3*T_SVD(handle_test[0],0.9)[1]))
    #test_cluster(handle_train, handle_test)
    #clusters_train(handle_train,handle_test)
    #ys.exit()
    #compare_3d_4d(handle_train,handle_test)
    #test_week_cluster(handle_train, handle_test,'var')
    plot_img(handle_train,ori_speeddata)
    sys.exit()

    #var_cluster(train_data,test_data)
    #test_week(train_data,test_data,sp_day)
    #fcm_ha(train_data,test_data)
    sys.exit()
    road_good(train_data)
    simMat = multi_sim(train_data)
    #scio.savemat('road_sim.mat', {'sim': simMat})

