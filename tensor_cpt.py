#encoding=utf-8
import sktensor
import matplotlib
import matplotlib.pyplot as plt
import sys
import os
import scipy.io as scio
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from random import random
from scipy.sparse import rand as sprand
from compt_methods import *
from road_similar import *
from kernel_kmeans import KernelKMeans


from sklearn.model_selection import train_test_split

#根据确定的数据维度大小获取数据，对于缺失部分补平均数
def get_tensor(mat_file,ori_file,size_=0):
    ori_data = scio.loadmat(mat_file)['Speed']
    if size_:
        p,q,r = size_
        ori_data = ori_data[:p,:q,:r]
    scio.savemat(ori_file,{'Speed':ori_data})
    return ori_data

#根据缺失率和原始数据进行随机缺失并保存缺少数据
def gene_rand_sparse(ori_data, miss_ratio,miss_file):
    data = ori_data.copy()
    dshape = np.shape(data)
    rand_ts = np.random.random_sample(dshape)
    zero_ts = np.zeros(dshape)
    data = data*(rand_ts>miss_ratio) + zero_ts*(rand_ts<=miss_ratio)
    #if not os.path.exists(miss_file):
    scio.savemat(miss_file,{'Speed':data})
    return data 

#根据缺失率和原始数据进行连续缺失并保存缺失数据
def gene_cont_sparse(ori_data,miss_ratio,miss_file):
    data = ori_data.copy()
    dshape = data.shape
    rand_ts = np.random.random_sample(dshape[:-1])
    S = np.rint(rand_ts+0.5-miss_ratio)
    W_cont = np.zeros(dshape)
    for k in range(dshape[2]):
        W_cont[:,:,k] = S[:,:]
    data = data*W_cont
    scio.savemat(miss_file,{'Speed':data})
    return data

gene_sparse = {'rand':gene_rand_sparse,'cont':gene_cont_sparse}

#获取缺失数据，缺失位置和真实缺失率
def get_sparsedata(miss_file):
    #miss_loc = []
    sparse_data = scio.loadmat(miss_file)['Speed']
    dshape = np.shape(sparse_data)
    W_miss = sparse_data==0
    true_miss_ratio = W_miss.sum()/sparse_data.size
    return sparse_data,W_miss,true_miss_ratio

#处理原始的缺失数据
def deal_orimiss(ori_data,shorten = False):
    p,q,r = ori_data.shape
    zero_mat = np.zeros((p,q,r))
    sparse_data = (ori_data>1)*ori_data+(ori_data<=1)*zero_mat
    W = sparse_data>0
    W_miss = np.where(sparse_data<=1)
    M_pos = np.where(W_miss)
    if not shorten:
        for i in range(len(W_miss[0])):
            pos1,pos2,pos3 = W_miss[0][i],W_miss[1][i],W_miss[2][i]
            neigh_info = []
            for n2 in (1,-1):
                for n3 in (1,-1):
                    try:
                        temp = sparse_data[pos1,pos2+n2,pos3+n3]
                        neigh_info.append(temp)
                    except:
                        pass
            if sum(neigh_info)>0:
                sparse_data[pos1,pos2,pos3] = sum(neigh_info)/(np.array(neigh_info)>0).sum()
        return sparse_data,W
            
    Arr = [set(arr.tolist()) for arr in M_pos]
    Arr_len = [len(arr) for arr in Arr]
    Arr_short = Arr_len.index(min(Arr_len))
    sparse_data = np.delete(sparse_data,list(Arr[Arr_short]),Arr_short)
    return sparse_data,W

#用奇异值比例来确定秩
def tk_rank(data,thre = 0.9):
    rank_set = [0,0,0]
    for i in range(3):
        if np.isnan(data).all():
            print(np.isnan(data).sum())
            sys.exit()
        if np.isnan(data).all():
            print(np.isinf(data).sum())
            sys.exit()
        U,sigma,VT = scipy.linalg.svd(dtensor(data).unfold(i),0)
        for r in range(len(sigma)):
            if sum(sigma[:r])/sum(sigma) > thre:
                rank_set[i] = r
                break
    return rank_set

#预填充，这里采用同一线圈同一时段不同日期的平均数填充
def pre_impute(sparse_data,W,day_axis=1,bias_bool = False):
    if not bias_bool:
        pos = np.where(W==False)
        for p in range(len(pos[0])):
            i,j,k = pos[0][p],pos[1][p],pos[2][p]
            if day_axis == 0:
                if (sparse_data[:,j,k]>0).sum()>0:
                    sparse_data[i,j,k] = np.sum(sparse_data[:,j,k])/(sparse_data[:,j,k]>0).sum()
                else:
                    sparse_data[i,j,k] = np.sum(sparse_data[i,:,:])/(sparse_data[i,:,:]>0).sum()
            elif day_axis == 1:
                if (sparse_data[i,:,k]>0).sum()>0:
                    sparse_data[i,j,k] = np.sum(sparse_data[i,:,k])/(sparse_data[i,:,k]>0).sum()
                else:
                    sparse_data[i,j,k] = np.sum(sparse_data[:,j,:])/(sparse_data[:,j,:]>0).sum()
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
    return sparse_data

#求rmse,mape,mae,rse
def rmse_mape_rse(est_data,ori_data,W):
    S = np.shape(est_data)
    W_miss = (W==False)
    diff_data = np.zeros(S)+W_miss*(est_data-ori_data)
    #diff_data = ori_data-est_data
    rmse = float((np.sum(diff_data**2)/W_miss.sum())**0.5)
    mre_mat=np.zeros_like(est_data)
    mre_mat[W_miss]=np.abs((est_data[W_miss]-ori_data[W_miss])/ori_data[W_miss])
    mape = float(np.sum(mre_mat)/W_miss.sum())
    rse = float(np.sum(diff_data**2)**0.5/np.sum(ori_data[W_miss]**2)**0.5)
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
    RM_dict,MA_dict,RS_dict,MP_dict = {},{},{},{}
    rt_dict = {}
    miss_list = []
    eva_dict = {'rmse':RM_dict,'mae':MA_dict,'mape':MP_dict}
    for i in range(8):
        miss_ratio = round(0.1*(i+1),2)

        miss_path = data_dir+'miss_'+str(miss_ratio)+''.join(['_'+str(ch) for ch in data_size])+'.mat'
        if not os.path.exists(miss_path):
            gene_rand_sparse(ori_speeddata,miss_ratio,miss_path)
        miss_data,W_miss,tm_ratio = get_sparsedata(miss_path)
        W = (W_miss==False)
        miss_list.append(int(tm_ratio * 100))
        #预填充
        miss_data = pre_impute(miss_data,W,False)
        rW = W|(ori_W==False)

        #参数
        p = 0.7
        K = 100     #iterations
        F_thre = 1e-5  #F_norm convergence threshold

        #halrtc
        time_s = time.time()
        lou = 1 / T_SVD(miss_data, p)[0][0]
        est_halrtc = halrtc_cpt(miss_data,lou,F_thre,K,W,alpha=[1/3,1/3,1/3])
        time_e = time.time()
        rm, mp, rs, ma = rmse_mape_rse(est_halrtc,ori_speeddata,rW)
        km = 'HaLRTC'
        if km not in RM_dict:
            RM_dict[km], MP_dict[km], RS_dict[km], MA_dict[km] = [], [], [], []
            rt_dict[km] = []
        RM_dict[km].append(rm)
        MA_dict[km].append(ma)
        MP_dict[km].append(mp)
        RS_dict[km].append(rs)
        rt_dict[km].append(time_e - time_s)

        #Kmeans+halrtc
        time_s = time.time()
        K_n = 4   #cluster_num
        est_kmeans = Kmeans_ha(miss_data, W, K_n, K, F_thre, p)
        time_e = time.time()
        rm,mp,rs,ma = rmse_mape_rse(est_kmeans,ori_speeddata,rW)
        km = 'HaLRTC_CSP'
        if km not in RM_dict:
            RM_dict[km],MP_dict[km],RS_dict[km],MA_dict[km] = [],[],[],[]
            rt_dict[km] = []
        RM_dict[km].append(rm)
        MA_dict[km].append(ma)
        MP_dict[km].append(mp)
        RS_dict[km].append(rs)
        rt_dict[km].append(time_e-time_s)

        #STD
        time_s = time.time()
        ap,lm,thre = 2e-10,0.05,0.1
        est_STD = STD_cpt(miss_data, W, thre, ap, lm, p)
        time_e = time.time()
        rm, mp, rs, ma = rmse_mape_rse(est_STD, ori_speeddata, rW)
        km = 'STD'
        if km not in RM_dict:
            RM_dict[km], MP_dict[km], RS_dict[km], MA_dict[km] = [], [], [], []
            rt_dict[km] = []
        RM_dict[km].append(rm)
        MA_dict[km].append(ma)
        MP_dict[km].append(mp)
        RS_dict[km].append(rs)
        rt_dict[km].append(time_e - time_s)

        #BPCA
        time_s = time.time()
        est_BPCA = BPCA_cpt(miss_data, p)
        time_e = time.time()
        rm, mp, rs, ma = rmse_mape_rse(est_BPCA, ori_speeddata, rW)
        km = 'BPCA'
        if km not in RM_dict:
            RM_dict[km], MP_dict[km], RS_dict[km], MA_dict[km] = [], [], [], []
            rt_dict[km] = []
        RM_dict[km].append(rm)
        MA_dict[km].append(ma)
        MP_dict[km].append(mp)
        RS_dict[km].append(rs)
        rt_dict[km].append(time_e - time_s)
        
    eva_dict = {'RMSE':RM_dict,'MAE':MA_dict,'MRE':MP_dict,'Run_Time':rt_dict}
    metric_dict = {'RMSE':'km/h', 'MAE':'km/h', 'MRE':'%', 'Run_Time':'s'}
    eva_Ylim = {'RMSE':[2,10],'MAE':[0,5],'MRE':[5,20],'Run_Time':[0,5000]}
    shape = ['r--o','r--*','r--x','r--^','r--s','r--D']
    MK = ['o','o','*','*','x','x']
    CR = ['r','b','y','r','b','y']
    for eva in eva_dict:
        #fig = plt.figure()
        #ax = fig.add_subplot(1,1,1)
        fw = open('compare_mr_'+'_'+eva+'.txt','w')
        fw.write('Missing Rate (%):' + ','.join(miss_list) + '\n')
        plt.xlabel('Missing Rate (%)')
        plt.ylabel(eva+' ('+metric_dict[eva]+')')
        # xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        #        [r'$-pi$', r'$-pi/2$', r'$0$', r'$+pi/2$', r'$+pi$'])
        nl = 0
        for method in eva_dict[eva]:
            plt.plot(miss_list,eva_dict[eva][method],color=CR[nl],marker=MK[nl],label='$'+method+'$')
            fw.write('eva:' + ','.join(eva_dict[eva][method])+'\n')
            nl += 1
        plt.legend(loc='best')
        plt.savefig(img_dir+'compare_mr_'+'_'+eva+'.png')
        plt.close()
        fw.close()
         
    return 0

def tkcp_res():
    miss_set = [0.01,0.05,0.1,0.15,0.2,0.25,0.3]
    fw = open('./占有率对比实验结果.txt','w')
    for miss_ratio in miss_set:
        fw.write('缺失率'+str(miss_ratio)+':\n')
        miss_path = data_dir+'miss_'+'_'.join([str(ch) for ch in data_size])+'.mat'
        gene_rand_sparse(ori_speeddata,miss_ratio,miss_path)
        miss_data,miss_pos,tm_ratio = get_sparsedata(miss_path)
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


def svd_vary(sparse_data):
    ds = sparse_data.shape
    SG = np.zeros((max(ds),3))
    for i in range(len(ds)):
        A = dtensor(sparse_data).unfold(i)
        # U,sigma,VT = np.linalg.svd(A,0)
        # SG[:len(sigma),i] = sigma
        for j in range(ds[i]):
            SG[j,i] = np.var(A[j])
        plt.plot(list(range(max(ds))),SG[:,i],'r--o',label='$'+str(i)+'$')
        break
    plt.legend(loc=1)
    plt.savefig(img_dir+'svd_vary.png')
    plt.close()
    return

def test_partmiss(ori_data,ori_W,sparse_data,W,new_mr):
    lou, K, conv_thre, fb = 3e-3, 100, 1e-5, 0.85
    est1 = halrtc_cpt(sparse_data, lou, conv_thre, K, W, fb)
    eva1 = rmse_mape_rse(est1, ori_data, (W | (ori_W == False)))
    print(eva1)
    ds = sparse_data.shape
    rand_ts = np.random.rand(ds[0], ds[1], ds[2])
    new_data = (rand_ts<new_mr)*np.zeros(ds)+sparse_data*(rand_ts>=new_mr)
    new_data[W] = sparse_data[W]
    W_new = new_data>0
    new_data = pre_impute(new_data,W_new)
    est2 = halrtc_cpt(new_data, lou, conv_thre, K, W_new, fb)
    eva2 = rmse_mape_rse(est2, ori_data, (W | (ori_W == False)))
    print(eva2)
    return

#比较三维张量和四维张量的填充结果
def compare_3d_4d(ori_speeddata,miss_data):
    '''
    weeks = data_size[1] // 7
    Nori_data = np.zeros((data_size[0], weeks, 7, data_size[2]))
    Nmiss_data = np.zeros_like(Nori_data)
    N_W = np.zeros_like(Nmiss_data)
    Nori_W = np.zeros_like(Nori_data)
    for i in range(data_size[1]):
        if i >= weeks * 7:
            break
        Nori_data[:, i // 7, i % 7, :] = ori_speeddata[:, i, :]
        Nori_W[:, i // 7, i % 7, :] = ori_W[:, i, :]
        Nmiss_data[:, i // 7, i % 7, :] = miss_data[:, i, :]
        N_W[:, i // 7, i % 7, :] = W[:, i, :]
    print(Nmiss_data.shape)
    time0 = time.time()
    est_halrtc = halrtc_cpt(Nmiss_data, 1e-3, 1e-4, 100, N_W, 0)
    time1 = time.time()
    print('4d_halrtc:', rmse_mape_rse(est_halrtc, Nori_data, N_W))
    print('4d_time', str(time1 - time0) + 's')
    '''
    time1 = time.time()
    lou = 1 / T_SVD(miss_data, 0.7)[0][0]
    print(lou)
    est_halrtc = halrtc_cpt(miss_data, lou, 1e-4, 100, W)
    time2 = time.time()
    print('3d_halrtc:', rmse_mape_rse(est_halrtc, ori_speeddata, (W | (ori_W == False))))
    print('3d_time', str(time2 - time1) + 's')
    return

def test_PPCA(data):
    ppca = pca.ppca.PPCA()
    ppca.fit(data)
    y = ppca.transform_infers()
    print(y.shape)
    return y

if __name__ == '__main__':
    data_dir = './data/'
    img_dir = './img_test/'
    mat_path = '/home/qiushye/2013_east/2013_east_speed.mat'
    #数据：日期数，线圈数量，时间间隔数
    #data_size = (60,80,144) 
    #data_size = (15,35,288)
    data_size = (30,20,72)
    #广州数据

    ori_speeddata = scio.loadmat('../GZ_data/speed_tensor.mat')['tensor']
    #train_sim(ori_speeddata[:,:30,:])
    #assign_group(ori_speeddata[:,:30,:])
    #simMat = multi_sim(ori_speeddata[:,:30,:])
    #scio.savemat('road_sim.mat',{'sim':simMat})
    #sys.exit()
    #print(np.var(ori_speeddata))
    ori_speeddata = ori_speeddata#[:20,:30,:40]#.swapaxes(0,2)
    shorten = False
    ori_speeddata,ori_W = deal_orimiss(ori_speeddata,shorten)
    #ori_speeddata_train = ori_speeddata[:30]
    #ori_speeddata,ori_W = ori_speeddata[30:],ori_W[30:]
    data_size = np.shape(ori_speeddata)
    print((ori_W==False).sum())
    print(data_size)

    compare_mr(ori_speeddata,ori_W)
    sys.exit()
    miss_ratio = 0.2
    miss_path = data_dir+'miss_'+str(round(miss_ratio,1))+'_'+'_'.join([str(ch) for ch in data_size])+'.mat'
    #miss_path = data_dir+'cont_miss_'+'_'.join([str(ch) for ch in data_size])+'.mat'
    if not os.path.exists(miss_path):
        gene_rand_sparse(ori_speeddata,miss_ratio,miss_path)
    #gene_cont_sparse(ori_speeddata,miss_ratio,miss_path)
    miss_data,W_miss,tm_ratio = get_sparsedata(miss_path)
    print('true_miss:',tm_ratio)
    W = miss_data>0
    #W1 = miss_data==0
    rank_set = [0,0,0]
    data_shape = np.shape(ori_speeddata)
    miss_data = pre_impute(miss_data,W)
    print('pre_impute:', rmse_mape_rse(miss_data, ori_speeddata, W | (ori_W == False)))
    #compare_3d_4d(ori_speeddata,miss_data)
    #tucker_cpt(miss_data,[50,20,50],W)
    # est_PPCA = PPCA_cpt(miss_data)
    est_BPCA = BPCA_cpt(miss_data, 0.7)
    print(rmse_mape_rse(est_BPCA,ori_speeddata,W))
    #est_Km = Kmeans_ha(miss_data,W,3,100,1e-4,0.7)
    #print(rmse_mape_rse(est_Km, ori_speeddata, W))
    # time0 = time.time()
    # est_STD = STD_cpt(miss_data, W, alpha=2e-10, lm=0.05, threshold=0.1)
    # time1 = time.time()
    # print('ori_STD:', rmse_mape_rse(est_STD, ori_speeddata, (W | (ori_W == False))))
    # print('ori_time', str(time1 - time0) + 's')
    #svd_vary(miss_data)
    '''
    
    SD = dtensor(miss_data)
    SGA = []
    for i in range(3):
        U,sigma,VT = np.linalg.svd(SD.unfold(i),0)
        SGA.append(sum(sigma))
    for j in range(3):
        print(SGA[j]/sum(SGA))
    '''

    sys.exit()
    #est_partmiss(ori_speeddata, ori_W, miss_data, W, 0.04)
    #u,sigma,vt = np.linalg.svd(dtensor(miss_data).unfold(1),0)
    #print(sigma[0],sigma[0]/sum(sigma))
    #sys.exit()
    lou = 1e-3
    K = 100
    conv = 1e-5


    halrtc_para = [3e-3,100,1e-5]
    [lou,K,conv_thre] = halrtc_para
    time0 = time.time()
    #est_halrtc = halrtc_cpt(miss_data, 1.3e-3, 1e-4, 100, W, 0)
    time1 = time.time()
    #print('ori_halrtc:', rmse_mape_rse(est_halrtc, ori_speeddata, (W | (ori_W == False))))
    print('ori_time', str(time1- time0) + 's')
    K_n = 2
    labels = SC_1(miss_data, 6, K_n, axis=0)
    est_SC = cluster_ha(labels, miss_data, W, K_n, halrtc_para, axis=0)
    time_e = time.time()
    print('sc_est:',rmse_mape_rse(est_SC, ori_speeddata, W|(ori_W==False)))
    clr_assign,K_n = road_Kmeans(miss_data,ori_W,K_n,W,axis=0,method='cos')
    est_kmeans = cluster_ha(clr_assign,miss_data,W,K_n,halrtc_para,axis=0)
    print('kmeans_est:',rmse_mape_rse(est_kmeans,ori_speeddata,W|(ori_W==False)))
    time2 = time.time()
    print('kmeans_time:',time2-time1,'s')
    sys.exit()
    cr = range(30)
    weekday,weekend = [],[]
    week_dict = {}
    for i in range(61):
        if i%7 not in week_dict:
            week_dict[i%7] = []
        week_dict[i%7].append(i)
        if i%7 < 5:
            weekday.append(i)
        else:
            weekend.append(i)

    est_week = np.zeros_like(miss_data)
    '''
    for wd in sorted(week_dict.keys()):
        est_halrtc = halrtc_cpt(miss_data[:,week_dict[wd],:],lou,conv_thre,K,W[:,week_dict[wd],:],fb)
        est_week[:,week_dict[wd],:] = est_halrtc
        print('halrtc:',rmse_mape_rse(est_halrtc,ori_speeddata[:,week_dict[wd],:],(W|(ori_W==False))[:,week_dict[wd],:]))
    '''

    est_halrtc = halrtc_cpt(miss_data[:, weekday, :], lou, conv_thre, K, W[:, weekday, :], fb)
    est_week[:, weekday, :] = est_halrtc
    est_halrtc = halrtc_cpt(miss_data[:, weekend, :], lou, conv_thre, K, W[:, weekend, :], fb)
    est_week[:, weekend, :] = est_halrtc
    print(rmse_mape_rse(est_week,ori_speeddata,W|(ori_W==False)))

    time3 = time.time()
    print('ha_time:',time3-time2,'s')
    sys.exit()
    w_dis = [0.1,0.8,0.7]
    K_n = 3
    svd_list = [0.01,0.1,1]
    var_list = [0.1,1,10]
    mean_list = [0.1,1,10]
    rmse_list,mape_list,mae_list = [],[],[]
    for var_para in var_list:
        for svd_para in svd_list:
            for mean_para in mean_list:
                w_dis = [var_para,svd_para,mean_para]
                labels = SC_1(miss_data,6,K_n,w_dis,method='multi',axis=0)
                est_SC = cluster_ha(labels,miss_data,W,K_n,halrtc_para,axis=0)
                rmse,mape,rse,mae = rmse_mape_rse(est_SC,ori_speeddata,W|(ori_W==False))
                rmse_list.append(rmse)
                mape_list.append(mape)
                mae_list.append(mae)
                print('---------')
                print(var_para,svd_para,mean_para)
                print(rmse,mape,mae)
    sys.exit()
    fig = plt.figure()
    plt.subplot(221)
    plt.plot(var_list,rmse_list)
    plt.subplot(222)
    plt.plot(var_list,mape_list)
    plt.subplot(223)
    plt.plot(var_list,mae_list)
    plt.savefig(img_dir+"SC_para.png")
    plt.close()
    print('SC_est:',rmse_mape_rse(est_SC,ori_speeddata,W|(ori_W==False)))
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
    #sys.exit()
    est_cp = cp_cpt(miss_data,15,W)
    RMSE_cp,MAPE_cp,RSE_cp,MAE_cp = rmse_mape_rse(est_cp,ori_speeddata,W)
    print('RMSE_cp,MAE_cp,MAPE_cp',RMSE_cp,MAE_cp,MAPE_cp)
    sys.exit()
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
