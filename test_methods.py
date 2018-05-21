#encoding=utf-8
from tensor_cpt import *
from compt_methods import *

def road_good(data_,axis=0):
    data = data_.copy()
    data, ori_W1 = deal_orimiss(data)
    ds = data.shape
    #ds_1 = np.array(ds)
    mr = 0.2
    miss_path1 = data_dir + 'miss_' + str(mr) + '_' + '_'.join([str(ch) for ch in ds]) + '.mat'
    if not os.path.exists(miss_path1):
        gene_rand_sparse(data, mr, miss_path1)
    miss_data1, W_, tr = get_sparsedata(miss_path1)
    W1 = miss_data1 > 0
    miss_data1 = pre_impute(miss_data1, W1)
    lou, K, conv_thre, fb = 3e-3, 100, 1e-5, 0.85
    est_ori = halrtc_cpt(miss_data1, lou, conv_thre, K, W1, fb)
    eva_ori = rmse_mape_rse(est_ori, data, (W1 | (ori_W1 == False)))
    good_part = []
    good_list = ['good_road.txt','good_day.txt','good_period.txt']
    for i in range(ds[axis]):
        slice1 = list(ds)
        slice1[axis] = i
        est1 = halrtc_cpt(miss_data1[slice1], lou, conv_thre, K, W1[slice1], fb)
        eva1 = rmse_mape_rse(est1, data[slice1], (W1 | (ori_W1 == False))[slice1])
        if eva1[0] < eva_ori[0] and eva1[1] < eva_ori[1]:
            good_part.append(i)
    if not os.path.exists(good_list[axis]):
        fw = open(good_list[axis],'w')
        fw.write(','.join(list(map(str,good_part)))+'\n')
        fw.close()
    return good_part


if __name__ == '__main__':
    data_dir = './data/'
    img_dir = './img_test/'
    #mat_path = '/home/qiushye/2013_east/2013_east_speed.mat'
    # 广州数据
    ori_speeddata = scio.loadmat('../GZ_data/speed_tensor.mat')['tensor']
    # train_sim(ori_speeddata[:,:30,:])
    #assign_group(ori_speeddata[:, :30, :])
    road_good(ori_speeddata[:,:30,:])
    simMat = multi_sim(ori_speeddata[:, :30, :])
    #scio.savemat('road_sim.mat', {'sim': simMat})
    sys.exit()