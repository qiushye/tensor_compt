from tensor_cpt import *

def compare_C(ori_speeddata,ori_W):
    miss_ratio = 0.2
    miss_path = data_dir + 'miss_' + str(round(miss_ratio,1)) + ''.join(['_' + str(ch) for ch in data_size]) + '.mat'
    if not os.path.exists(miss_path):
        gene_rand_sparse(ori_speeddata, miss_ratio, miss_path)
    miss_data, W_miss, tm_ratio = get_sparsedata(miss_path)
    W = (W_miss == False)
    # 预填充
    miss_data = pre_impute(miss_data, W, False)
    rW = W | (ori_W == False)

    p = 0.7
    K = 100  # iterations
    F_thre = 1e-4  # F_norm convergence threshold

    RM_dict, MA_dict, RS_dict, MP_dict = {}, {}, {}, {}
    rt_dict = {}
    C_list = list(range(2,11))
    for C in C_list:
        time_s = time.time()
        K_n = C  # cluster_num
        est_kmeans = Kmeans_ha(miss_data, W, K_n, K, F_thre, p)
        time_e = time.time()
        rm, mp, rs, ma = rmse_mape_rse(est_kmeans, ori_speeddata, rW)
        km = 'HaLRTC_CSP'
        if km not in RM_dict:
            RM_dict[km], MP_dict[km], RS_dict[km], MA_dict[km] = [], [], [], []
            rt_dict[km] = []
        RM_dict[km].append(rm)
        MA_dict[km].append(ma)
        MP_dict[km].append(mp)
        RS_dict[km].append(rs)
        rt_dict[km].append(round(time_e - time_s,1))

    eva_dict = {'RMSE': RM_dict, 'MAE': MA_dict, 'MRE': MP_dict, 'Run_Time': rt_dict}
    metric_dict = {'RMSE': 'km/h', 'MAE': 'km/h', 'MRE': '%', 'Run_Time': 's'}
    nl = 0
    fw = open('compare_C.txt', 'w')
    fw.write('Clustering Number:' + ','.join(list(map(str, C_list))) + '\n')
    for eva in eva_dict:
        plt.plot(C_list,eva_dict[eva][km],color='r',marker='o')
        plt.xlabel('Clustering Number')
        plt.ylabel(eva + ' (' + metric_dict[eva] + ')')
        plt.legend(loc='best')
        plt.savefig(img_dir + 'compare_C_' + '_' + eva + '.png')
        plt.close()
        fw.write(eva + ':\n')
        fw.write(','.join(list(map(str, eva_dict[eva][km]))) + '\n')
    fw.close()
    return

def compare_alpha(ori_speeddata,ori_W):
    alpha_list = [0.5,0.3,0.2]
    for i in range(3):
        ori_speeddata = ori_speeddata.swapaxes(0,i)
        ori_W = ori_W.swapaxes(0,i)
        data_size = ori_speeddata.shape
        miss_ratio = 0.2
        miss_path = data_dir + 'miss_' + str(round(miss_ratio, 1)) + ''.join(['_' + str(ch) for ch in data_size]) + '.mat'
        if not os.path.exists(miss_path):
            gene_rand_sparse(ori_speeddata, miss_ratio, miss_path)
        miss_data, W_miss, tm_ratio = get_sparsedata(miss_path)
        W = (W_miss == False)
        # 预填充
        miss_data = pre_impute(miss_data, W, False)
        rW = W | (ori_W == False)

        p = 0.7
        K = 100  # iterations
        F_thre = 1e-4  # F_norm convergence threshold
        time_s = time.time()
        K_n = 4  # cluster_num
        #est_kmeans = Kmeans_ha(miss_data, W, K_n, K, F_thre, p,alpha_list)
        lou = 1 / T_SVD(miss_data, 0.7)[0][0]
        est_halrtc = halrtc_cpt(miss_data, lou, 1e-5, 100, W, alpha_list)
        time_e = time.time()
        rm, mp, rs, ma = rmse_mape_rse(est_halrtc, ori_speeddata, rW)
        print(i,rm,mp,rs,ma)
        print(i,str(time_e-time_s)+'s')
    return


def compare_p(ori_speeddata, ori_W):
    RM_dict, MA_dict, RS_dict, MP_dict = {}, {}, {}, {}
    rt_dict = {}
    miss_list = []
    eva_dict = {'rmse': RM_dict, 'mae': MA_dict, 'mape': MP_dict}
    miss_ratio = 0.2

    miss_path = data_dir + 'miss_' + str(miss_ratio) + ''.join(['_' + str(ch) for ch in data_size]) + '.mat'
    if not os.path.exists(miss_path):
        gene_rand_sparse(ori_speeddata, miss_ratio, miss_path)
    miss_data, W_miss, tm_ratio = get_sparsedata(miss_path)
    W = (W_miss == False)
    miss_list.append(round(tm_ratio * 100, 1))
    # 预填充
    miss_data = pre_impute(miss_data, W, False)
    rW = W | (ori_W == False)
    p_list = []
    for i in range(4):
        # 参数
        p = 0.6+i*0.1
        p_list.append(round(p,1))
        K = 100  # iterations
        F_thre = 1e-5  # F_norm convergence threshold
        print('----'+str(i)+'----')
        '''
        # halrtc
        time_s = time.time()
        lou = 1 / T_SVD(miss_data, p)[0][0]
        est_halrtc = halrtc_cpt(miss_data, lou, F_thre, K, W, alpha=[1 / 3, 1 / 3, 1 / 3])
        time_e = time.time()
        rm, mp, rs, ma = rmse_mape_rse(est_halrtc, ori_speeddata, rW)
        km = 'HaLRTC'
        if km not in RM_dict:
            RM_dict[km], MP_dict[km], RS_dict[km], MA_dict[km] = [], [], [], []
            rt_dict[km] = []
        RM_dict[km].append(rm)
        MA_dict[km].append(ma)
        MP_dict[km].append(mp)
        RS_dict[km].append(rs)
        rt_dict[km].append(round(time_e - time_s, 1))

        # Kmeans+halrtc
        time_s = time.time()
        K_n = 4  # cluster_num
        est_kmeans = Kmeans_ha(miss_data, W, K_n, K, F_thre, p)
        time_e = time.time()
        rm, mp, rs, ma = rmse_mape_rse(est_kmeans, ori_speeddata, rW)
        km = 'HaLRTC-CSP'
        if km not in RM_dict:
            RM_dict[km], MP_dict[km], RS_dict[km], MA_dict[km] = [], [], [], []
            rt_dict[km] = []
        RM_dict[km].append(rm)
        MA_dict[km].append(ma)
        MP_dict[km].append(mp)
        RS_dict[km].append(rs)
        rt_dict[km].append(round(time_e - time_s, 1))
        '''
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
        rt_dict[km].append(round(time_e - time_s,1))
        '''
        # BPCA
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
        rt_dict[km].append(round(time_e - time_s, 1))
        '''

    eva_dict = {'RMSE': RM_dict, 'MAE': MA_dict, 'MRE': MP_dict, 'Run_Time': rt_dict}
    metric_dict = {'RMSE': 'km/h', 'MAE': 'km/h', 'MRE': '%', 'Run_Time': 's'}
    eva_Ylim = {'RMSE': [2, 10], 'MAE': [0, 5], 'MRE': [5, 20], 'Run_Time': [0, 5000]}
    shape = ['r--o', 'r--*', 'r--x', 'r--^', 'r--s', 'r--D']
    MK = ['o', 'o', '*', '*', 'x', 'x']
    CR = ['r', 'b', 'y', 'r', 'b', 'y']

    fw = open('compare_methods_p' + '.txt', 'w')
    fw.write('methods:' + ','.join(list(eva_dict['RMSE'].keys())) + '\n')
    fw.write('Truncated rate:' + ','.join(list(map(str, p_list))) + '\n')
    for eva in eva_dict:
        # fig = plt.figure()
        # ax = fig.add_subplot(1,1,1)
        plt.xlabel('Truncated rate')
        plt.ylabel(eva + ' (' + metric_dict[eva] + ')')
        # xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        #        [r'$-pi$', r'$-pi/2$', r'$0$', r'$+pi/2$', r'$+pi$'])
        nl = 0
        fw.write(eva + ':\n')
        for method in eva_dict[eva]:
            plt.plot(p_list, eva_dict[eva][method], color=CR[nl], marker=MK[nl], label='$' + method + '$')
            fw.write(','.join(list(map(str, eva_dict[eva][method]))) + '\n')
            nl += 1
        plt.legend(loc='best')
        plt.savefig(img_dir + 'compare_methods_p_' + '_' + eva + '.png')
        plt.close()
    fw.close()

    return 0

def result_plot(file_path):
    RM_dict, MA_dict, RS_dict, MP_dict = {}, {}, {}, {}
    rt_dict = {}
    eva_dict = {'RMSE': RM_dict, 'MAE': MA_dict, 'MRE': MP_dict, 'Run_Time': rt_dict}
    #miss_list = [10,20,30,40,50,60,70,80]
    p_list = [0.6,0.7,0.8,0.9]
    f = open(file_path,'r')
    row = f.readline().strip().split(',')
    row[0] = row[0].split(':')[-1]
    method_list = row
    print(method_list)
    f.readline()
    for line in f.readlines():
        while '\n' in line:
            line = line.strip()
        row = line.split(',')
        if len(row) == 1:
            m = 0
            eva = row[0][:-1]
            continue
        print(eva)
        print(row)
        eva_dict[eva][method_list[m]] = [float(s) for s in row]
        m += 1

    metric_dict = {'RMSE': 'km/h', 'MAE': 'km/h', 'MRE': '%', 'Run_Time': 's'}
    eva_Ylim = {'RMSE': [2, 10], 'MAE': [0, 5], 'MRE': [5, 20], 'Run_Time': [0, 5000]}
    shape = ['r--o', 'r--*', 'r--x', 'r--^', 'r--s', 'r--D']
    MK = ['o', 'o', '*', '*', 'x', 'x']
    CR = ['r', 'b', 'y', 'r', 'b', 'y']

    for eva in eva_dict:
        # fig = plt.figure()
        # ax = fig.add_subplot(1,1,1)
        plt.xlabel('Truncated Singular Value Rate (%)')
        #plt.xlabel('Missing Rate (%)')
        plt.ylabel(eva + ' (' + metric_dict[eva] + ')')
        # xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        #        [r'$-pi$', r'$-pi/2$', r'$0$', r'$+pi/2$', r'$+pi$'])
        nl = 0
        for method in eva_dict[eva]:
            #plt.plot(miss_list, eva_dict[eva][method], color=CR[nl], marker=MK[nl], label='$' + method + '$')
            plt.plot(p_list, eva_dict[eva][method], color=CR[nl], marker=MK[nl], label='$' + method + '$')
            nl += 1
        plt.legend(loc='best')
        plt.savefig(img_dir + 'compare_methods_p_' + eva + '.png')
        plt.close()
    return

if __name__ == '__main__':
    data_dir = './data/'
    img_dir = './img_test/'
    mat_path = '/home/qiushye/2013_east/2013_east_speed.mat'
    #广州数据

    ori_speeddata = scio.loadmat('../GZ_data/60days_tensor.mat')['tensor']
    shorten = False
    print((ori_speeddata==0).sum()/ori_speeddata.size)
    ori_speeddata,ori_W = deal_orimiss(ori_speeddata,shorten)
    data_size = np.shape(ori_speeddata)
    #compare_methods(ori_speeddata,ori_W)
    #compare_PI(ori_speeddata,ori_W)
    #compare_C(ori_speeddata, ori_W)
    #compare_alpha(ori_speeddata,ori_W)
    #compare_p(ori_speeddata,ori_W)
    result_plot('compare_methods_p_1.txt')
    sys.exit()
