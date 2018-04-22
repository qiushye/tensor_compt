#encoding=utf-8
import sktensor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import sys
import os
import scipy.io as scio
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from random import random
from scipy.sparse import rand as sprand
from compt_methods import *
from road_similar import *

#根据确定的数据维度大小获取数据，对于缺失部分补平均数
def get_tensor(mat_file,ori_file,size_=0):
    ori_data = scio.loadmat(mat_file)['Speed']
    if size_:
        p,q,r = size_
        ori_data = ori_data[:p,:q,:r]
    scio.savemat(ori_file,{'Speed':ori_data})
    return ori_data

#根据缺失率和原始数据进行随机缺失并保存缺少数据
def gene_rand_sparse(ori_data, miss_radio,miss_file):
    data = ori_data.copy()
    dshape = np.shape(data)
    rand_ts = np.random.rand(dshape[0],dshape[1],dshape[2])
    zero_ts = np.zeros(dshape)
    data = data*(rand_ts>miss_radio) + zero_ts*(rand_ts<=miss_radio)
    scio.savemat(miss_file,{'Speed':data})
    return data 

#根据缺失率和原始数据进行连续缺失并保存缺失数据
def gene_cont_sparse(ori_data,miss_radio,miss_file):
    data = ori_data.copy()
    dshape = data.shape
    rand_ts = np.random.rand(dshape[0],dshape[1])
    S = np.rint(rand_ts+0.5-miss_radio)
    W_cont = np.zeros(dshape)
    for k in range(dshape[2]):
        W_cont[:,:,k] = S[:,:]
    data = data*W_cont
    scio.savemat(miss_file,{'Speed':data})
    return data


#获取缺失数据，缺失位置和真实缺失率
def get_sparsedata(miss_file):
    #miss_loc = []
    sparse_data = scio.loadmat(miss_file)['Speed']
    dshape = np.shape(sparse_data)
    W_miss = sparse_data==0
    '''
    for i in range(dshape[0]):
        for j in range(dshape[1]):
            for k in range(dshape[2]):
                if sparse_data[i,j,k] == 0:
                    miss_loc.append((i,j,k))
    '''
    true_miss_radio = W_miss.sum()/sparse_data.size
    return sparse_data,W_miss,true_miss_radio

#将数据标准化，除以线圈维度的2范数
def norm_data(data_input):
    if not data_input.any():
        print('no data input')
        return data_input
    data = data_input.copy()
    mode1,mode2,mode3 = np.shape(data)
    for i in range(mode1):
        for j in range(mode2):
            norm_mode1 = dtensor(data[i,j,:]).norm()
            data[i,j,:] /= norm_mode1
    return data

#处理原始的缺失数据
def deal_orimiss(ori_data,shorten = True):
    p,q,r = ori_data.shape
    zero_mat = np.zeros((p,q,r))
    sparse_data = (ori_data>1)*ori_data+(ori_data<=1)*zero_mat
    W = sparse_data>0
    W_miss = np.where(sparse_data<=1)
    ori_missloc = []
    M_pos = np.where(W_miss)
    if not shorten:
        for i in range(len(W_miss[0])):
            pos1,pos2,pos3 = W_miss[0][i],W_miss[1][i],W_miss[2][i]
            neigh_info = []
            for n2 in (1,-1):
                for n3 in (1,-1):
                    try:
                        neigh_info.append(sparse_data[pos1,pos2+n2,pos3+n3])
                    except:
                        pass
            #neigh_info = [sparse_data[pos1,pos2-1,pos3],sparse_data[pos1,pos2+1,pos3],
                    #sparse_data[pos1,pos2,pos3-1],sparse_data[pos1,pos2,pos3+1]]
            sparse_data[pos1,pos2,pos3] = sum(neigh_info)/(np.array(neigh_info)>0).sum()
            ori_missloc.append((pos1,pos2,pos3))
        return sparse_data,W
            
    Arr = [set(arr.tolist()) for arr in M_pos]
    Arr_len = [len(arr) for arr in Arr]
    Arr_short = Arr_len.index(min(Arr_len))
    sparse_data = np.delete(sparse_data,list(Arr[Arr_short]),Arr_short)
    return sparse_data,W


#预填充，这里采用同一线圈同一时段不同日期的平均数填充
def pre_impute(sparse_data,W,bias_bool = False):
    if not bias_bool:
        pos = np.where(W==False)
        for p in range(len(pos[0])):
            i,j,k = pos[0][p],pos[1][p],pos[2][p]
            sparse_data[i,j,k] = np.sum(sparse_data[i,:,k])/(sparse_data[i,:,k]>0).sum()
        return sparse_data
    b,b1 = {},{}
    sp = np.shape(sparse_data)
    impute_data = sparse_data.copy()
    mean = np.sum(impute_data)/(impute_data>0).sum()
    for n in range(3):
        b[n] = np.random.uniform(0,0.1,sp[n])
        b1[n] = np.zeros(sp[n])
    seta,miu = 1e-5,800
    sum_F = sum([np.sum(b[n]**2) for n in range(3)])
    J = 1/2*np.sum(W*(sparse_data-impute_data)**2)+miu/2*sum_F
    ite = 0
    while ite < 100:
        ite += 1
        J_pre = J
        for i in range(sp[0]):
            b1[0][i] = (1-seta*miu)*b[0][i]+seta*np.sum(W[i,:,:]*(sparse_data-impute_data)[i,:,:])
        for j in range(sp[1]):
            b1[1][j] = (1-seta*miu)*b[1][j]+seta*np.sum(W[:,j,:]*(sparse_data-impute_data)[:,j,:])
        for k in range(sp[2]):
            b1[2][k] = (1-seta*miu)*b[2][k]+seta*np.sum(W[:,:,k]*(sparse_data-impute_data)[:,:,k])
        if sum([np.sum((b1[n]-b[n])**2)**0.5 for n in range(3)]) < 0.001:
            pass
        for n in range(3): 
            b[n] = b1[n].copy()
        for i in range(sp[0]):
            for j in range(sp[1]):
                for k in range(sp[2]):
                    impute_data[i,j,k] = mean+b[0][i]+b[1][j]+b[2][k]
        sum_F = sum([np.sum(b[n]**2) for n in range(3)])
        J = 1/2*np.sum(W*(sparse_data-impute_data)**2)+miu/2*sum_F
        if abs(J-J_pre) < 1:
            break
    print(np.sum(W*(sparse_data-impute_data)**2))
    sparse_data[W==False] = mean+b[0][i]+b[1][j]+b[2][k]
    '''
    for _set in np.where(W==0):
        sparse_data[_set] = mean+b[0][i]+b[1][j]+b[2][k]
    '''
    return sparse_data

#求rmse,mape和rse
def rmse_mape_rse(est_data,ori_data,W):
    S = np.shape(est_data)
    diff_data = np.zeros(S)
    '''
    for i in range(S[0]):
        for j in range(S[1]):
            for k in range(S[2]):
                if (i,j,k) in miss_loc:
                    diff_data[i,j,k] = est_data[i,j,k]-ori_data[i,j,k]
                else:
                    diff_data[i,j,k] = 0
    '''
    W_miss = (W==False)
    diff_data = np.zeros(S)+W_miss*(est_data-ori_data)
    #diff_data = ori_data-est_data
    rmse = float((np.sum(diff_data**2)/W_miss.sum())**0.5)
    mre_mat=np.zeros_like(est_data)
    mre_mat[W_miss]=np.abs((est_data[W_miss]-ori_data[W_miss])/ori_data[W_miss])
    mape = float(np.sum(mre_mat)/W_miss.sum())
    rse = float(np.sum(diff_data**2)**0.5/np.sum(ori_data**2)**0.5)
    mae = float(np.sum(np.abs(diff_data))/W_miss.sum())
    return round(rmse,4),round(mape,4),round(rse,4),round(mae,4)

def show_img(X,RMSE_list,MAE_list,name_list):
    for i in range(len(name_list)):
        plt.plot(X,RMSE_list[i],'--o')
        plt.savefig(img_dir+'rmse_'+name_list[i])
        plt.close()
        plt.plot(X,MAE_list[i],'--o')
        plt.savefig(img_dir+'mae_'+name_list[i])
        plt.close()
    return 0

def compare_iter(ori_speeddata,miss_data,miss_pos,W):
    sp = np.shape(miss_data)
    rank_set = [0,0,0]
    main_rate = 0.9
    '''
    RMSE_tk_list,MAE_tk_list = [],[]
    tk_range_list = []
    for count in range(40):
        main_rate = 0.6+count*(1-0.6)/40
        tk_range_list.append(main_rate)
        for i in range(3):
            U,sigma,VT = scipy.linalg.svd(dtensor(miss_data).unfold(i),0)
            for r in range(len(sigma)):
                if sum(sigma[:r])/sum(sigma) > main_rate:
                    rank_set[i] = r
                    break
        est_tucker = tucker_cpt(miss_data,rank_set,W)
        RMSE_tk,MAPE_tk,RSE_tk,MAE_tk = rmse_mape_rse(est_tucker,ori_speeddata,W)
        RMSE_tk_list.append(RMSE_tk)
        MAE_tk_list.append(MAE_tk)
    show_img(tk_range_list,[RMSE_tk_list],[MAE_tk_list],['tk'])
            
    cp_rank = 15
    cp_range_list = list(range(int(0.25*min(sp)),int(0.75*min(sp))))
    RMSE_cp_list,MAE_cp_list = [],[]
    for cp_rank in cp_range_list:
        est_cp = cp_cpt(miss_data,cp_rank,W)
        RMSE_cp,MAPE_cp,RSE_cp,MAE_cp = rmse_mape_rse(est_cp,ori_speeddata,W)
        RMSE_cp_list.append(RMSE_cp)
        MAE_cp_list.append(MAE_cp)
    show_img(cp_range_list,[RMSE_cp_list],[MAE_cp_list],['cp'])
    '''
    alpha = [1/3,1/3,1/3]
    beta = [0.1,0.1,0.1]
    beta1 = beta.copy()
    gama = [2,2,2]
    lou = 1e-3
    K = 100
    conv = 1e-4
    conv_list = np.arange(1e-5,1e-3,5e-5)
    K_list = [50+10*count for count in range(16)]
    RMSE_lrtc_list,MAE_lrtc_list = [],[]
    RMSE_silrtc_list,MAE_silrtc_list = [],[]
    RMSE_halrtc_list,MAE_halrtc_list = [],[]
    '''
    range_list = K_list
    for K in K_list:
        est_lrtc = lrtc_cpt(miss_data,beta,alpha,gama,conv,K,W)
        RMSE_lrtc,MAPE_lrtc,RSE_lrtc,MAE_lrtc = rmse_mape_rse(est_lrtc,ori_speeddata,W)
        RMSE_lrtc_list.append(RMSE_lrtc)
        MAE_lrtc_list.append(MAE_lrtc)
        est_silrtc = silrtc_cpt(miss_data,alpha,beta1,conv,K)
        RMSE_silrtc,MAPE_silrtc,RSE_silrtc,MAE_silrtc = rmse_mape_rse(est_silrtc,ori_speeddata,W)
        RMSE_silrtc_list.append(RMSE_silrtc)
        MAE_silrtc_list.append(MAE_silrtc)
        est_halrtc = halrtc_cpt(miss_data,lou,conv,K,W)
        RMSE_halrtc,MAPE_halrtc,RSE_halrtc,MAE_halrtc = rmse_mape_rse(est_halrtc,ori_speeddata,W)
        RMSE_halrtc_list.append(RMSE_halrtc)
        MAE_halrtc_list.append(MAE_halrtc)
    RMSE_tc = [RMSE_lrtc_list,RMSE_silrtc_list,RMSE_halrtc_list]
    MAE_tc = [MAE_lrtc_list,MAE_silrtc_list,MAE_halrtc_list]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(range_list,RMSE_lrtc_list,'r--^',label='lrtc')
    ax.plot(range_list,RMSE_silrtc_list,'r--s',label='silrtc')
    ax.plot(range_list,RMSE_halrtc_list,'r--D',label='halrtc')
    ax.legend(loc='best')
    plt.savefig(img_dir+'compare_ite_rmse.png')
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(range_list,MAE_lrtc_list,'r--^',label='lrtc')
    ax.plot(range_list,MAE_silrtc_list,'r--s',label='silrtc')
    ax.plot(range_list,MAE_halrtc_list,'r--D',label='halrtc')
    ax.legend(loc='best')
    plt.savefig(img_dir+'compare_ite_mae.png')
    plt.close()
    '''
    RMSE_lrtc_list,MAE_lrtc_list = [],[]
    RMSE_silrtc_list,MAE_silrtc_list = [],[]
    RMSE_halrtc_list,MAE_halrtc_list = [],[]
    range_list = conv_list
    for conv in conv_list:
        est_lrtc = lrtc_cpt(miss_data,beta,alpha,gama,conv,K,W)
        RMSE_lrtc,MAPE_lrtc,RSE_lrtc,MAE_lrtc = rmse_mape_rse(est_lrtc,ori_speeddata,W)
        RMSE_lrtc_list.append(RMSE_lrtc)
        MAE_lrtc_list.append(MAE_lrtc)
        est_silrtc = silrtc_cpt(miss_data,alpha,beta1,conv,K)
        RMSE_silrtc,MAPE_silrtc,RSE_silrtc,MAE_silrtc = rmse_mape_rse(est_silrtc,ori_speeddata,W)
        RMSE_silrtc_list.append(RMSE_silrtc)
        MAE_silrtc_list.append(MAE_silrtc)
        est_halrtc = halrtc_cpt(miss_data,lou,conv,K,W)
        RMSE_halrtc,MAPE_halrtc,RSE_halrtc,MAE_halrtc = rmse_mape_rse(est_halrtc,ori_speeddata,W)
        RMSE_halrtc_list.append(RMSE_halrtc)
        MAE_halrtc_list.append(MAE_halrtc)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(range_list,RMSE_lrtc_list,'r--^',label='lrtc')
    ax.plot(range_list,RMSE_silrtc_list,'r--s',label='silrtc')
    ax.plot(range_list,RMSE_halrtc_list,'r--D',label='halrtc')
    ax.legend(loc='best')
    plt.savefig(img_dir+'compare_conv_rmse.png')
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(range_list,MAE_lrtc_list,'r--^',label='lrtc')
    ax.plot(range_list,MAE_silrtc_list,'r--s',label='silrtc')
    ax.plot(range_list,MAE_halrtc_list,'r--D',label='halrtc')
    ax.legend(loc='best')
    plt.savefig(img_dir+'compare_conv_mae.png')
    plt.close()


    #show_img(K_list,RMSE_tc,MAE_tc,['lrtc','silrtc','halrtc'])
    return 0

def compare_mr(ori_speeddata,ori_W):
    R_pre_l,R_tk_l,R_cp_l = [],[],[]
    R_lr_l,R_silr_l,R_halr_l = [],[],[]
    MA_pre_l,MA_tk_l,MA_cp_l = [],[],[]
    MA_lr_l,MA_silr_l,MA_halr_l = [],[],[]
    MP_pre_l,MP_tk_l,MP_cp_l = [],[],[]
    MP_lr_l,MP_silr_l,MP_halr_l = [],[],[]
    R_l,MA_l,MP_l = [],[],[]
    miss_list = []
    for i in range(4):
        miss_ratio = 0.1*(i+1)
        miss_list.append(miss_ratio)
        miss_path = data_dir+'miss_'+'_'.join([str(ch) for ch in data_size])+'_test.mat'
        #if not os.path.exists(miss_path):
        gene_rand_sparse(ori_speeddata,miss_ratio,miss_path)
        miss_data,miss_pos,tm_radio = get_sparsedata(miss_path)
        W = miss_data>0
        miss_data = pre_impute(miss_data,W,False)
        print(np.mean(ori_speeddata),np.mean(miss_data))
        rW = W|ori_W
        RMSE_pre,MAPE_pre,RSE_pre,MAE_pre = rmse_mape_rse(miss_data,ori_speeddata,rW)
        R_pre_l.append(RMSE_pre)
        MA_pre_l.append(MAE_pre)
        MP_pre_l.append(MAPE_pre)
        rank_set = [0,0,0]
        cp_a = min(ori_speeddata.shape)//2
        for i in range(3):
            U,sigma,VT = scipy.linalg.svd(dtensor(miss_data).unfold(i),0)
            for r in range(len(sigma)):
                if sum(sigma[:r])/sum(sigma) > 0.9:
                    rank_set[i] = r
                    break
        est_tucker = tucker_cpt(miss_data,rank_set,W)
        RMSE_tk,MAPE_tk,RSE_tk,MAE_tk = rmse_mape_rse(est_tucker,ori_speeddata,rW)
        
        R_tk_l.append(RMSE_tk)
        MA_tk_l.append(MAE_tk)
        MP_tk_l.append(MAPE_tk)
        est_cp = cp_cpt(miss_data,cp_a,W)
        RMSE_cp,MAPE_cp,RSE_cp,MAE_cp = rmse_mape_rse(est_cp,ori_speeddata,rW)
        R_cp_l.append(RMSE_cp)
        MA_cp_l.append(MAE_cp)
        MP_cp_l.append(MAPE_cp)
        
        alpha = [1.0/3,1.0/3,1.0/3]
        beta = [0.1,0.1,0.1]
        beta1 = [0.1,0.1,0.1]
        gama = [2,2,2]
        lou = 1e-3
        K = 100
        conv = 1e-4
        '''        
        est_lrtc = lrtc_cpt(miss_data,alpha,beta,gama,conv,K)
        RMSE_lrtc,MAPE_lrtc,RSE_lrtc,MAE_lrtc = rmse_mape_rse(est_lrtc,ori_speeddata,rW)
        #print(RMSE_lrtc)
        #sys.exit()
        R_lr_l.append(RMSE_lrtc)
        MA_lr_l.append(MAE_lrtc)
        MP_lr_l.append(MAPE_lrtc)
        est_silrtc = silrtc_cpt(miss_data,alpha,beta1,conv,K)
        RMSE_silrtc,MAPE_silrtc,RSE_silrtc,MAE_silrtc = rmse_mape_rse(est_silrtc,ori_speeddata,rW)
        R_silr_l.append(RMSE_silrtc)
        MA_silr_l.append(MAE_silrtc)
        MP_silr_l.append(MAPE_silrtc)
        '''
        est_halrtc = halrtc_cpt(miss_data,lou,conv,K,W)
        RMSE_halrtc,MAPE_halrtc,RSE_halrtc,MAE_halrtc = rmse_mape_rse(est_halrtc,ori_speeddata,rW)
        print('rmse:',RMSE_halrtc)
        R_halr_l.append(RMSE_halrtc)
        MA_halr_l.append(MAE_halrtc)
        MP_halr_l.append(MAPE_halrtc)
        
    R_l = [R_pre_l,R_tk_l,R_cp_l,R_lr_l,R_silr_l,R_halr_l]
    MA_l = [MA_pre_l,MA_tk_l,MA_cp_l,MA_lr_l,MA_silr_l,MA_halr_l]
    MP_l = [MP_pre_l,MP_tk_l,MP_cp_l,MP_lr_l,MP_silr_l,MP_halr_l]
    eva_dict = {'rmse':R_l,'mae':MA_l,'mape':MP_l}
    name_l = ['pre','tucker','cp','lrtc','silrtc','halrtc']
    shape = ['r--o','r--*','r--x','r--^','r--s','r--D']
    for eva in eva_dict:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        for i in range(6):
            print(name_l[i],eva_dict[eva][i])
            ax.plot(miss_list,eva_dict[eva][i],shape[i],label=name_l[i])
        ax.legend(loc='best')
        plt.savefig(img_dir+'compare_mr_'+eva+'_shorten'*shorten+'.png')
        plt.close()
         
    return 0

def halrtc_cmp(ori_speeddata,ori_W):
    R_halr_l,MA_halr_l,MP_halr_l = [],[],[]
    R_halr_fb,MA_halr_fb,MP_halr_fb = [],[],[]
    miss_list = []
    for i in range(4):
        miss_ratio = 0.1*(i+1)
        miss_list.append(miss_ratio)
        miss_path = data_dir+'miss_'+'_'.join([str(ch) for ch in data_size])+'_test.mat'
        #if not os.path.exists(miss_path):
        gene_rand_sparse(ori_speeddata,miss_ratio,miss_path)
        miss_data,miss_pos,tm_radio = get_sparsedata(miss_path)
        W = miss_data>0
        miss_data = pre_impute(miss_data,W,False)
        print(np.mean(ori_speeddata),np.mean(miss_data))
        rW = W|ori_W
        print(rW.sum()-W.sum())
        alpha = [1.0/3,1.0/3,1.0/3]
        beta = [0.1,0.1,0.1]
        beta1 = [0.1,0.1,0.1]
        gama = [2,2,2]
        lou = 1e-3
        K = 100
        conv = 1e-4
        fb = 0.85
        est_halrtc = halrtc_cpt(miss_data,lou,conv,K,W,fb)
        RMSE_halrtc,MAPE_halrtc,RSE_halrtc,MAE_halrtc = rmse_mape_rse(est_halrtc,ori_speeddata,rW)
        R_halr_l.append(RMSE_halrtc)
        MA_halr_l.append(MAE_halrtc)
        MP_halr_l.append(MAPE_halrtc)
        print(MA_halr_l)
        
        est_halrtc_fb = halrtc_cpt(miss_data,lou,conv,K,W,fb,True)
        RMSE_halrtc,MAPE_halrtc,RSE_halrtc,MAE_halrtc = rmse_mape_rse(est_halrtc_fb,ori_speeddata,rW)
        R_halr_fb.append(RMSE_halrtc)
        MA_halr_fb.append(MAE_halrtc)
        MP_halr_fb.append(MAPE_halrtc)
        print(MA_halr_fb)
        print(np.mean(est_halrtc[rW==False]-est_halrtc_fb[rW==False]))
        
    eva_dict = ['rmse','mae','mape']
    plt.plot(miss_list,R_halr_l,'r--o',label='halrtc')
    plt.plot(miss_list,R_halr_fb,'r--*',label='halrtc_fb')
    plt.savefig(img_dir+'compare_mr_rmse_fb_cmp.png')
    plt.close()
    plt.plot(miss_list,MA_halr_l,'r--o',label='halrtc')
    plt.plot(miss_list,MA_halr_fb,'r--*',label='halrtc_fb')
    plt.savefig(img_dir+'compare_mr_mae_fb_cmp.png')
    plt.close()
    plt.plot(miss_list,MP_halr_l,'r--o',label='halrtc')
    plt.plot(miss_list,MP_halr_fb,'r--*',label='halrtc_fb')
    plt.savefig(img_dir+'compare_mr_mape_fb_cmp.png')
    plt.close()
    return 0

def tkcp_res():
    miss_set = [0.01,0.05,0.1,0.15,0.2,0.25,0.3]
    fw = open('./占有率对比实验结果.txt','w')
    for miss_radio in miss_set:
        fw.write('缺失率'+str(miss_radio)+':\n')
        miss_path = data_dir+'miss_'+'_'.join([str(ch) for ch in data_size])+'.mat'
        gene_rand_sparse(ori_speeddata,miss_radio,miss_path)
        miss_data,miss_pos,tm_radio = get_sparsedata(miss_path)
        W = miss_data>0
        miss_data = pre_impute(miss_data,W)
        rank_set = [0,0,0]
        for i in range(3):
            U,sigma,VT = scipy.linalg.svd(dtensor(miss_data).unfold(i),0)
            for r in range(len(sigma)):
                if sum(sigma[:r])/sum(sigma) > 0.9:
                    rank_set[i] = r
                    break

        est_tucker = tucker_cpt(miss_data,rank_set,W)
        RMSE_tk,MAPE_tk,RSE_tk,MAE_tk = rmse_mape_rse(est_tucker,ori_speeddata,miss_pos)
        fw.write('tucker填充: RMSE,MAPE,MAE='+str(RMSE_tk)+','+str(MAPE_tk)+','+str(MAE_tk)+'\n')
        est_cp = cp_cpt(miss_data,6,W)
        RMSE_cp,MAPE_cp,RSE_cp,MAE_cp = rmse_mape_rse(est_cp,ori_speeddata,miss_pos)
        fw.write('cp填充: RMSE,MAPE,MAE='+str(RMSE_cp)+','+str(MAPE_cp)+','+str(MAE_cp)+'\n')
    fw.close()
    return 0

def test_kmeans(sparse_data,K_n,ori_W,W,halrtc_para):
    Clr_center,clr_assign = road_Kmeans(sparse_data,ori_W,K_n,W)
    Clr_mat = {i:[] for i in range(K_n)}
    for i in range(sparse_data.shape[0]):
        Clr_mat[clr_assign[i,0]].append(i)
    [lou,K,conv_thre,fb] = halrtc_para
    est_data = np.zeros_like(sparse_data)
    for j in range(K_n):
        m_data = sparse_data[(clr_assign[:,0]==j)]
        Wm = W[(clr_assign[:,0]==j)]
        temp_data = halrtc_cpt(m_data,lou,conv_thre,K,Wm,fb)
        est_data[Clr_mat[j]] = temp_data
    return est_data

def test_SC(sparse_data,KNN_k,cluster_num,halrtc_para):
    labels = SC_1(sparse_data,KNN_k,cluster_num)
    Clr_mat = {i:[] for i in range(cluster_num)}
    for i in range(len(labels)):
        Clr_mat[labels[i]].append(i)
    print([len(Clr_mat[i]) for i in Clr_mat])
    [lou,K,conv_thre,fb] = halrtc_para
    est_data = np.zeros_like(sparse_data)
    for j in range(cluster_num):
        m_data = sparse_data[labels == j]
        Wm = W[labels == j]
        #temp_data = halrtc_cpt(m_data,lou,conv_thre,K,Wm,fb)
        temp_data = cp_cpt(m_data,len(m_data)//3,Wm)
        est_data[Clr_mat[j]] = temp_data
    return est_data

if __name__ == '__main__':
    data_dir = './data/'
    img_dir = './img_test/'
    mat_path = '/home/qiushye/2013_east/2013_east_speed.mat'
    #数据：日期数，线圈数量，时间间隔数
    #data_size = (60,80,144) 
    #data_size = (15,35,288)
    data_size = (30,20,72)
    '''
    ori_path = data_dir+'ori_'+'_'.join([str(ch) for ch in data_size])+'.mat'
    if not os.path.exists(ori_path):
        ori_speeddata = get_tensor(mat_path,ori_path,data_size)
    else:
        ori_speeddata = scio.loadmat(ori_path)['Speed']
    '''
    #ori_path = './Occudata.mat'
    #ori_speeddata = scio.loadmat(ori_path)['Occu']
    #广州数据
    
    #ori_speeddata = scio.loadmat('/home/qiushye/GZ_data/speed_tensor.mat')['tensor']
    ori_speeddata = scio.loadmat('../GZ_data/speed_tensor.mat')['tensor']
    shorten = False
    ori_speeddata,ori_W = deal_orimiss(ori_speeddata,shorten)
    data_size = np.shape(ori_speeddata)
    print((ori_W==False).sum())
    print(data_size)
    #sys.exit()
    #compare_mr(ori_speeddata,ori_missloc)
    
    #sys.exit()
    miss_radio = 0.2
    miss_path = data_dir+'miss_'+'_'.join([str(ch) for ch in data_size])+'.mat'
    #miss_path = data_dir+'cont_miss_'+'_'.join([str(ch) for ch in data_size])+'.mat'
    gene_rand_sparse(ori_speeddata,miss_radio,miss_path)
    #gene_cont_sparse(ori_speeddata,miss_radio,miss_path)
    miss_data,W,tm_radio = get_sparsedata(miss_path)
    W = miss_data>0
    W1 = miss_data==0
    rank_set = [0,0,0]
    data_shape = np.shape(ori_speeddata)
    #ori_speeddata = norm_data(ori_speeddata)
    #miss_data = norm_data(miss_data)
    miss_data = pre_impute(miss_data,W)
    print((W==False).sum(),((W==False)&ori_W).sum())
    #road_Kmeans(miss_data,ori_W,4,W)
    #labels = SC(miss_data,ori_W,W,4)
    labels = SC_1(miss_data,10,3)
    #sys.exit()
    print('pre_impite:',rmse_mape_rse(miss_data,ori_speeddata,W|(ori_W==False)))
    halrtc_para = [1e-3,100,1e-5,0.85]
    [lou,K,conv_thre,fb] = halrtc_para    
    est_SC = test_SC(miss_data,5,2,halrtc_para)
    print('SC_est:',rmse_mape_rse(est_SC,ori_speeddata,W|(ori_W==False)))
    est_cp = cp_cpt(miss_data,15,W)
    RMSE_cp,MAPE_cp,RSE_cp,MAE_cp = rmse_mape_rse(est_cp,ori_speeddata,W)
    print('RMSE_cp,MAE_cp,MAPE_cp',RMSE_cp,MAE_cp,MAPE_cp)
    #sys.exit()
    #est_kmeans = test_kmeans(miss_data,3,ori_W,W,halrtc_para)
    #print('kmeans_est:',rmse_mape_rse(est_kmeans,ori_speeddata,W))
    #est_halrtc = halrtc_cpt(miss_data,lou,conv_thre,K,W,fb)
    #print('halrtc:',rmse_mape_rse(est_halrtc,ori_speeddata,W|(ori_W==False)))
    #compare_iter(ori_speeddata,miss_data,miss_pos,W)
    #halrtc_cmp(ori_speeddata,ori_W)
    #sys.exit()
    '''
    rates = 0.5 + np.array(range(8))*0.05
    est_multi = multi_tucker(miss_data,rates,miss_pos,W)
    est_data = sum([rate * est_multi[rate] for rate in est_multi])/sum(est_multi.keys())
    print([rmse_mape_rse(est,ori_speeddata,miss_pos) for est in est_multi.values()])
    print('rmse_mape_rse_mae',rmse_mape_rse(est_data,ori_speeddata,miss_pos))
    sys.exit()
    '''
    for i in range(3):
        U,sigma,VT = scipy.linalg.svd(dtensor(miss_data).unfold(i),0)
        for r in range(len(sigma)):
            if sum(sigma[:r])/sum(sigma) > 0.75:
                rank_set[i] = r
                break
    print(rank_set) 
    est_tucker = tucker_cpt(miss_data,rank_set,W)
    RMSE_tk,MAPE_tk,RSE_tk,MAE_tk = rmse_mape_rse(est_tucker,ori_speeddata,W|(ori_W==False))
    print('RMSE_tk,MAE_tk,MAPE_tk',RMSE_tk,MAE_tk,MAPE_tk)
    sys.exit()
    est_cp = cp_cpt(miss_data,15,W)
    RMSE_cp,MAPE_cp,RSE_cp,MAE_cp = rmse_mape_rse(est_cp,ori_speeddata,W)
    print('RMSE_cp,MAE_cp,MAPE_cp',RMSE_cp,MAE_cp,MAPE_cp)
    alpha = [1/3,1/3,1/3]
    beta = [0.1,0.1,0.1]
    beta1 = [0.1,0.1,0.1]
    gama = [2,2,2]
    lou = 1e-3
    K = 100
    conv = 1e-5
    #sys.exit()
    
    est_lrtc = lrtc_cpt(miss_data,beta,alpha,gama,conv,K,W)
    RMSE_lrtc,MAPE_lrtc,RSE_lrtc,MAE_lrtc = rmse_mape_rse(est_lrtc,ori_speeddata,W)
    print('RMSE_lrtc,MAE_lrtc',RMSE_lrtc,MAE_lrtc)
    #sys.exit()
    est_silrtc = silrtc_cpt(miss_data,alpha,beta1,conv,K,W)
    RMSE_silrtc,MAPE_silrtc,RSE_silrtc,MAE_silrtc = rmse_mape_rse(est_silrtc,ori_speeddata,W)
    print('RMSE_si,MAE_si',RMSE_silrtc,MAE_silrtc)
    
    est_halrtc = halrtc_cpt(miss_data,lou,conv,K,W)
    RMSE_halrtc,MAPE_halrtc,RSE_halrtc,MAE_halrtc = rmse_mape_rse(est_halrtc,ori_speeddata,W)
    print('RMSE_ha,MAE_ha,MAPE', (RMSE_halrtc,MAE_halrtc,MAPE_halrtc))
