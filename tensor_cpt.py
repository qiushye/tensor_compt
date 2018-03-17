#encoding='utf=8'
import sktensor
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

#根据确定的数据维度大小获取数据，对于缺失部分补平均数
def get_tensor(mat_file,ori_file,size_=0):
    ori_data = scio.loadmat(mat_file)['Speed']
    if size_:
        p,q,r = size_
        ori_data = ori_data[:p,:q,:r]
        for i in range(p):
            for j in range(q):
                for k in range(r):
                    if ori_data[i,j,k] == -1:
                        ori_data[i,j,k] = np.sum(ori_data[:,j,k])/(ori[:,j,k]>0).sum()

    scio.savemat(ori_file,{'Speed':ori_data})
    return ori_data

#根据缺失率和原始数据进行随机缺失并保存缺少数据
def gene_sparse(ori_data, miss_radio,miss_file):
    data = ori_data.copy()
    dshape = np.shape(data)
    rand_ts = np.random.rand(dshape[0],dshape[1],dshape[2])
    zero_ts = np.zeros(dshape)
    data = data*(rand_ts>miss_radio) + zero_ts*(rand_ts<=miss_radio)
    scio.savemat(miss_file,{'Speed':data})
    return 0 

#获取缺失数据，缺失位置和真实缺失率
def get_sparsedata(miss_file):
    miss_loc = []
    sparse_data = scio.loadmat(miss_file)['Speed']
    dshape = np.shape(sparse_data)
    for i in range(dshape[0]):
        for j in range(dshape[1]):
            for k in range(dshape[2]):
                if sparse_data[i,j,k] == 0:
                    miss_loc.append((i,j,k))
    true_miss_radio = len(miss_loc)/sparse_data.size
    return sparse_data,miss_loc,true_miss_radio

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

#预填充，这里采用同一线圈同一时段不同日期的平均数填充
def pre_impute(sparse_data,miss_loc,W,bias_bool = False):
    if not bias_bool:
        for _set in miss_loc:
            i,j,k = _set
            sparse_data[i,j,k] = np.sum(sparse_data[:,j,k])/(sparse_data[:,j,k]>0).sum()
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
    for _set in miss_loc:
        i,j,k = _set
        sparse_data[i,j,k] = mean+b[0][i]+b[1][j]+b[2][k]
    return sparse_data

#求rmse,mape和rse
def rmse_mape_rse(est_data,ori_data,miss_loc):
    diff_data = ori_data-est_data
    S = np.shape(diff_data)
    diff_sum = 0
    rmse = float((np.sum(diff_data**2)/len(miss_loc))**0.5)
    mre_mat=np.zeros_like(est_data)
    elig_ind=np.where(ori_data>0)
    mre_mat[elig_ind]=np.abs((est_data[elig_ind]-ori_data[elig_ind])/ori_data[elig_ind])
    mape = np.sum(mre_mat)/len(miss_loc)
    '''
    for i in range(S[0]):
        for j in range(S[1]):
            for k in range(S[2]):
                if diff_data[i,j,k] != 0 and ori_data[i,j,k] > 0:
                    diff_sum += abs(diff_data[i,j,k])/ori_data[i,j,k]
    mape = diff_sum*100/len(miss_loc)
    '''
    rse = float(np.sum(diff_data**2)**0.5/np.sum(ori_data**2)**0.5)
    mae = float(np.sum(np.abs(diff_data))/len(miss_loc))
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
            U,sigma,VT = np.linalg.svd(dtensor(miss_data).unfold(i))
            for r in range(len(sigma)):
                if sum(sigma[:r])/sum(sigma) > main_rate:
                    rank_set[i] = r
                    break
        est_tucker = tucker_cpt(miss_data,miss_pos,rank_set,W)
        RMSE_tk,MAPE_tk,RSE_tk,MAE_tk = rmse_mape_rse(est_tucker,ori_speeddata,miss_pos)
        RMSE_tk_list.append(RMSE_tk)
        MAE_tk_list.append(MAE_tk)
    show_img(tk_range_list,[RMSE_tk_list],[MAE_tk_list],['tk'])
            
    cp_rank = 15
    cp_range_list = list(range(int(0.25*min(sp)),int(0.75*min(sp))))
    RMSE_cp_list,MAE_cp_list = [],[]
    for cp_rank in cp_range_list:
        est_cp = cp_cpt(miss_data,miss_pos,cp_rank)
        RMSE_cp,MAPE_cp,RSE_cp,MAE_cp = rmse_mape_rse(est_cp,ori_speeddata,miss_pos)
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
        est_lrtc = lrtc_cpt(miss_data,miss_pos,beta,alpha,gama,conv,K,W)
        RMSE_lrtc,MAPE_lrtc,RSE_lrtc,MAE_lrtc = rmse_mape_rse(est_lrtc,ori_speeddata,miss_pos)
        RMSE_lrtc_list.append(RMSE_lrtc)
        MAE_lrtc_list.append(MAE_lrtc)
        est_silrtc = silrtc_cpt(miss_data,miss_pos,alpha,beta1,conv,K)
        RMSE_silrtc,MAPE_silrtc,RSE_silrtc,MAE_silrtc = rmse_mape_rse(est_silrtc,ori_speeddata,miss_pos)
        RMSE_silrtc_list.append(RMSE_silrtc)
        MAE_silrtc_list.append(MAE_silrtc)
        est_halrtc = halrtc_cpt(miss_data,miss_pos,lou,conv,K,W)
        RMSE_halrtc,MAPE_halrtc,RSE_halrtc,MAE_halrtc = rmse_mape_rse(est_halrtc,ori_speeddata,miss_pos)
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
        est_lrtc = lrtc_cpt(miss_data,miss_pos,beta,alpha,gama,conv,K,W)
        RMSE_lrtc,MAPE_lrtc,RSE_lrtc,MAE_lrtc = rmse_mape_rse(est_lrtc,ori_speeddata,miss_pos)
        RMSE_lrtc_list.append(RMSE_lrtc)
        MAE_lrtc_list.append(MAE_lrtc)
        est_silrtc = silrtc_cpt(miss_data,miss_pos,alpha,beta1,conv,K)
        RMSE_silrtc,MAPE_silrtc,RSE_silrtc,MAE_silrtc = rmse_mape_rse(est_silrtc,ori_speeddata,miss_pos)
        RMSE_silrtc_list.append(RMSE_silrtc)
        MAE_silrtc_list.append(MAE_silrtc)
        est_halrtc = halrtc_cpt(miss_data,miss_pos,lou,conv,K,W)
        RMSE_halrtc,MAPE_halrtc,RSE_halrtc,MAE_halrtc = rmse_mape_rse(est_halrtc,ori_speeddata,miss_pos)
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

def compare_mr(ori_speeddata):
    R_pre_l,R_tk_l,R_cp_l = [],[],[]
    R_lr_l,R_silr_l,R_halr_l = [],[],[]
    MA_pre_l,MA_tk_l,MA_cp_l = [],[],[]
    MA_lr_l,MA_silr_l,MA_halr_l = [],[],[]
    MP_pre_l,MP_tk_l,MP_cp_l = [],[],[]
    MP_lr_l,MP_silr_l,MP_halr_l = [],[],[]
    R_l,MA_l,MP_l = [],[],[]
    miss_list = []
    for i in range(4):
        miss_ratio = 0.05*(i+1)
        miss_list.append(miss_ratio)
        miss_path = data_dir+'miss_'+'_'.join([str(ch) for ch in data_size])+'_test.mat'
        #if not os.path.exists(miss_path):
        gene_sparse(ori_speeddata,miss_ratio,miss_path)
        miss_data,miss_pos,tm_radio = get_sparsedata(miss_path)
        W = miss_data>0
        miss_data = pre_impute(miss_data,miss_pos,W,False)
        RMSE_pre,MAPE_pre,RSE_pre,MAE_pre = rmse_mape_rse(miss_data,ori_speeddata,miss_pos)
        R_pre_l.append(RMSE_pre)
        MA_pre_l.append(MAE_pre)
        MP_pre_l.append(MAPE_pre)
        rank_set = [0,0,0]
        cp_a = min(ori_speeddata.shape)//2
        for i in range(3):
            U,sigma,VT = np.linalg.svd(dtensor(miss_data).unfold(i))
            for r in range(len(sigma)):
                if sum(sigma[:r])/sum(sigma) > 0.9:
                    rank_set[i] = r
                    break
        est_tucker = tucker_cpt(miss_data,miss_pos,rank_set,W)
        RMSE_tk,MAPE_tk,RSE_tk,MAE_tk = rmse_mape_rse(est_tucker,ori_speeddata,miss_pos)
        R_tk_l.append(RMSE_tk)
        MA_tk_l.append(MAE_tk)
        MP_tk_l.append(MAPE_tk)
        est_cp = cp_cpt(miss_data,miss_pos,cp_a)
        RMSE_cp,MAPE_cp,RSE_cp,MAE_cp = rmse_mape_rse(est_cp,ori_speeddata,miss_pos)
        R_cp_l.append(RMSE_cp)
        MA_cp_l.append(MAE_cp)
        MP_cp_l.append(MAPE_cp)
        
        alpha = [1/3,1/3,1/3]
        beta = [0.1,0.1,0.1]
        beta1 = beta.copy()
        gama = [2,2,2]
        lou = 1e-3
        K = 100
        conv = 1e-4
        est_lrtc = lrtc_cpt(miss_data,miss_pos,beta,alpha,gama,conv,K,W)
        RMSE_lrtc,MAPE_lrtc,RSE_lrtc,MAE_lrtc = rmse_mape_rse(est_lrtc,ori_speeddata,miss_pos)
        R_lr_l.append(RMSE_lrtc)
        MA_lr_l.append(MAE_lrtc)
        MP_lr_l.append(MAPE_lrtc)
        est_silrtc = silrtc_cpt(miss_data,miss_pos,alpha,beta1,conv,K)
        RMSE_silrtc,MAPE_silrtc,RSE_silrtc,MAE_silrtc = rmse_mape_rse(est_silrtc,ori_speeddata,miss_pos)
        R_silr_l.append(RMSE_silrtc)
        MA_silr_l.append(MAE_silrtc)
        MP_silr_l.append(MAPE_silrtc)
        est_halrtc = halrtc_cpt(miss_data,miss_pos,lou,conv,K,W)
        RMSE_halrtc,MAPE_halrtc,RSE_halrtc,MAE_halrtc = rmse_mape_rse(est_halrtc,ori_speeddata,miss_pos)
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
        for i in range(len(eva)):
            ax.plot(miss_list,eva_dict[eva][i],shape[i],label=name_l[i])
        ax.legend(loc='best')
        plt.savefig(img_dir+'compare_mr_'+eva+'.png')
        fig.close()
    '''
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(miss_list,R_pre_l,'r--o',label='pre')
    ax.plot(miss_list,R_tk_l,'r--*',label='tucker')
    ax.plot(miss_list,R_cp_l,'r--x',label='cp')
    ax.plot(miss_list,R_lr_l,'r--^',label='lrtc')
    ax.plot(miss_list,R_silr_l,'r--s',label='silrtc')
    ax.plot(miss_list,R_halr_l,'r--D',label='halrtc')
    ax.legend(loc='best')
    plt.savefig(img_dir+'compare_mr_rmse.png')
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(miss_list,MA_pre_l,'r--o',label='pre')
    ax.plot(miss_list,MA_tk_l,'r--*',label='tucker')
    ax.plot(miss_list,MA_cp_l,'r--x',label='cp')
    ax.plot(miss_list,MA_lr_l,'r--^',label='lrtc')
    ax.plot(miss_list,MA_silr_l,'r--s',label='silrtc')
    ax.plot(miss_list,MA_halr_l,'r--D',label='halrtc')
    ax.legend(loc='best')
    plt.savefig(img_dir+'compare_mr_mae.png')
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(miss_list,MA_pre_l,'r--o',label='pre')
    ax.plot(miss_list,MA_tk_l,'r--*',label='tucker')
    ax.plot(miss_list,MA_cp_l,'r--x',label='cp')
    ax.plot(miss_list,MA_lr_l,'r--^',label='lrtc')
    ax.plot(miss_list,MA_silr_l,'r--s',label='silrtc')
    ax.plot(miss_list,MA_halr_l,'r--D',label='halrtc')
    ax.legend(loc='best')
    plt.savefig(img_dir+'compare_mr_mae.png')
    plt.close()
    '''
    return 0

def tkcp_res():
    miss_set = [0.01,0.05,0.1,0.15,0.2,0.25,0.3]
    fw = open('./占有率对比实验结果.txt','w')
    for miss_radio in miss_set:
        fw.write('缺失率'+str(miss_radio)+':\n')
        miss_path = data_dir+'miss_'+'_'.join([str(ch) for ch in data_size])+'.mat'
        gene_sparse(ori_speeddata,miss_radio,miss_path)
        miss_data,miss_pos,tm_radio = get_sparsedata(miss_path)
        W = miss_data>0
        miss_data = pre_impute(miss_data,miss_pos,W)
        rank_set = [0,0,0]
        for i in range(3):
            U,sigma,VT = np.linalg.svd(dtensor(miss_data).unfold(i))
            for r in range(len(sigma)):
                if sum(sigma[:r])/sum(sigma) > 0.9:
                    rank_set[i] = r
                    break

        est_tucker = tucker_cpt(miss_data,miss_pos,rank_set,W)
        RMSE_tk,MAPE_tk,RSE_tk,MAE_tk = rmse_mape_rse(est_tucker,ori_speeddata,miss_pos)
        fw.write('tucker填充: RMSE,MAPE,MAE='+str(RMSE_tk)+','+str(MAPE_tk)+','+str(MAE_tk)+'\n')
        est_cp = cp_cpt(miss_data,miss_pos,6)
        RMSE_cp,MAPE_cp,RSE_cp,MAE_cp = rmse_mape_rse(est_cp,ori_speeddata,miss_pos)
        fw.write('cp填充: RMSE,MAPE,MAE='+str(RMSE_cp)+','+str(MAPE_cp)+','+str(MAE_cp)+'\n')
    fw.close()
    return 0


if __name__ == '__main__':
    data_dir = './data/'
    img_dir = './img_test/'
    mat_path = '/home/qiushye/2013_east/2013_east_speed.mat'
    #数据：日期数，线圈数量，时间间隔数
    #data_size = (60,80,144) 
    #data_size = (15,35,288)
    #data_size = (30,20,72)
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
    
    ori_speeddata = scio.loadmat('/home/qiushye/GZ_data/speed_tensor.mat')['tensor']
    data_size = np.shape(ori_speeddata)
    for i in range(data_size[0]):
        for j in range(data_size[1]):
            for k in range(data_size[2]):
                if ori_speeddata[i,j,k] == 0:
                    ori_speeddata[i,j,k] = np.sum(ori_speeddata[:,j,k])/(ori_speeddata[:,j,k]>0).sum()
    
    compare_mr(ori_speeddata)
    #tkcp_res()
    sys.exit()
    miss_radio = 0.1
    miss_path = data_dir+'miss_'+'_'.join([str(ch) for ch in data_size])+'.mat'
    gene_sparse(ori_speeddata,miss_radio,miss_path)
    miss_data,miss_pos,tm_radio = get_sparsedata(miss_path)
    W = miss_data>0
    W1 = miss_data==0
    rank_set = [0,0,0]
    data_shape = np.shape(ori_speeddata)
    #ori_speeddata = norm_data(ori_speeddata)
    #miss_data = norm_data(miss_data)
    miss_data = pre_impute(miss_data,miss_pos,W)
    #compare_iter(ori_speeddata,miss_data,miss_pos,W)
    #sys.exit()
    RMSE_pre,MAPE_pre,RSE_pre,MAE_pre = rmse_mape_rse(miss_data,ori_speeddata,miss_pos)
    print('RMSE_pre,MAE_pre,MAPE_pre',RMSE_pre,MAE_pre,MAPE_pre)
    
    for i in range(3):
        U,sigma,VT = np.linalg.svd(dtensor(miss_data).unfold(i))
        for r in range(len(sigma)):
            if sum(sigma[:r])/sum(sigma) > 0.9:
                rank_set[i] = r
                break
    print('ori_mean:',np.sum(ori_speeddata)/np.cumprod(data_size)[-1])
    
    est_tucker = tucker_cpt(miss_data,miss_pos,rank_set,W)
    RMSE_tk,MAPE_tk,RSE_tk,MAE_tk = rmse_mape_rse(est_tucker,ori_speeddata,miss_pos)
    print('RMSE_tk,MAE_tk,MAPE_tk',RMSE_tk,MAE_tk,MAPE_tk)
    est_cp = cp_cpt(miss_data,miss_pos,15)
    RMSE_cp,MAPE_cp,RSE_cp,MAE_cp = rmse_mape_rse(est_cp,ori_speeddata,miss_pos)
    print('RMSE_cp,MAE_cp,MAPE_cp',RMSE_cp,MAE_cp,MAPE_cp)
    alpha = [1/3,1/3,1/3]
    beta = [0.1,0.1,0.1]
    beta1 = beta.copy()
    gama = [2,2,2]
    lou = 1e-3
    K = 100
    conv = 1e-4
    '''
    est_lrtc = lrtc_cpt(miss_data,miss_pos,beta,alpha,gama,conv,K,W)
    RMSE_lrtc,MAPE_lrtc,RSE_lrtc,MAE_lrtc = rmse_mape_rse(est_lrtc,ori_speeddata,miss_pos)
    print('RMSE_lrtc,MAE_lrtc',RMSE_lrtc,MAE_lrtc)
    #sys.exit()
    est_silrtc = silrtc_cpt(miss_data,miss_pos,alpha,beta1,conv,K)
    RMSE_silrtc,MAPE_silrtc,RSE_silrtc,MAE_silrtc = rmse_mape_rse(est_silrtc,ori_speeddata,miss_pos)
    print('RMSE_si,MAE_si',RMSE_silrtc,MAE_silrtc)
    '''
    est_halrtc = halrtc_cpt(miss_data,miss_pos,lou,conv,K,W)
    RMSE_halrtc,MAPE_halrtc,RSE_halrtc,MAE_halrtc = rmse_mape_rse(est_halrtc,ori_speeddata,miss_pos)
    print('RMSE_ha,MAE_ha,MAPE',RMSE_halrtc,MAE_halrtc,MAPE_halrtc)
    
