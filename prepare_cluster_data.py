# 根据聚类结果将原始顶点重排,并按照簇团划分数据集，返回新的HF_data

# 1. 加载聚类结果

import numpy as np
import os

res = np.load("./dress02/Cluster_motion0_res.npz",allow_pickle=True)
def reorder_vertex(init_data,order_list):
    # init_data: [500,12273,3] == 第二个维度重排
    ordered_data = np.zeros_like(init_data)
    ordered_data[:,:,:] = init_data[:,order_list,:]
    return ordered_data

def get_class_label(y):
    # y:根据模型聚类的结果提取每种类别下对应的元素标签
    clusters = np.unique(y)
    print("Cluser num :{}".format(clusters.shape[0]))
    class_list = [0]
    max_num = 0
    min_num = 12273
    temp = [] # 原始顶点重排对应的新索引列表
    for cluster in clusters:
        row_ix = np.where(y == cluster)
        print(row_ix)
        class_list.append(row_ix[0].shape[0]+class_list[-1])
        max_num = max(max_num,row_ix[0].shape[0])
        min_num = min(min_num,row_ix[0].shape[0])
        for i in row_ix[0]:
            temp.append(i)
    return class_list,temp,max_num,min_num

data = res["data"]
affinity = res["affinity"]
spectral = res["spectral"]
kmeans = res["mbkmeans"]
agglo = res["agglo"]

affinity_index,affinity_order,max_aff,min_aff = get_class_label(affinity)  # [max,min] == [68,19]
spectral_index,spectral_order,max_spectral,min_spectral = get_class_label(spectral)  # [max,min] == [2255,910]
kmeans_index,kmeans_order,max_kmeans,min_kmeans = get_class_label(kmeans)  
agglo_index,agglo_order,max_agglo,min_agglo = get_class_label(agglo)  

# 2. 根据聚类结果对原始数据重排并保存 - 先将原始数据拷贝到path路径下
path = "./dress02/affinity_res//"
for i in os.listdir(path)[:]:
    if "npz" in i:
        tmp = np.load(os.path.join(path,i),allow_pickle=True)
        if (tmp["final_ground"].shape!=(500,12273,3)):
            continue
        cluster_res = reorder_vertex(tmp["final_ground"],affinity_order)
        # print(cluster_res[:,0,:] == tmp["final_ground"][:,2255,:]) # for test...
        np.savez(os.path.join(path,i),affinity_res = cluster_res) 

# 3.保存聚类基本信息
# np.savez(os.path.join("./dress02/affinity_res/","affinity_id"),index=affinity_index,order = affinity_order,dims=[max_aff,min_aff]) 
# np.savez(os.path.join("./dress02/spectral_res/","spectral_id"),index=spectral_index,order = spectral_order,dims=[max_spectral,min_spectral]) 
# np.savez(os.path.join("./dress02/kmeans_res/","kmeans_id"),index=kmeans_index,order = kmeans_order,dims=[max_kmeans,min_kmeans]) 
# np.savez(os.path.join("./dress02/agglo_res/","agglo_id"),index=agglo_index,order = agglo_order,dims=[max_agglo,min_agglo]) 



# 4.根据聚类结果对邻接矩阵进行分割[12273,12273] -> [clusters,max_nums,max_nums]
# 结果用adjacent_matrix保存到本地而不是edge_index的稀疏格式 -> 是为了保证每个类存储数据格式大小完全一致; 
# 不同类的边个数可能不同,导致edge_index[2,edge_nums]大小也不同
def cluster_adj(cluster_index,cluster_order,max_num,init_adj=None):
    from torch_geometric.utils import get_laplacian,to_dense_adj,dense_to_sparse
    import torch
    from torch_geometric.transforms import FaceToEdge
    from torch_geometric.data import Data
    from src.obj_parser import Mesh_obj

    cloth = Mesh_obj("assets/dress02/garment.obj")
    data = FaceToEdge()(Data(num_nodes=cloth.v.shape[0],face=torch.from_numpy(cloth.f.astype(int).transpose() - 1).long()))
    init_adj = to_dense_adj(data.edge_index)[0].numpy() #numpy格式类型
    reorder_res = init_adj[cluster_order,:][:,cluster_order]
    
    cluster_num = len(cluster_index) - 1
    final_adj = np.zeros((cluster_num,max_num,max_num),dtype=np.int16)
    for i in range(cluster_num):
        final_adj[i,:cluster_index[i+1]-cluster_index[i],:cluster_index[i+1]-cluster_index[i]] = reorder_res[cluster_index[i]:cluster_index[i+1],cluster_index[i]:cluster_index[i+1]]
    return final_adj,reorder_res
spectral_adj,reorder0 = cluster_adj(spectral_index,spectral_order,max_spectral)
affinity_adj,reorder1 = cluster_adj(affinity_index,affinity_order,max_aff)
kmeans_adj,reorder2 = cluster_adj(kmeans_index,kmeans_order,max_kmeans)
agglo_adj,reorder3 = cluster_adj(agglo_index,agglo_order,max_agglo)

# np.save(os.path.join("./dress02/spectral_res/","spectral_adj"),spectral_adj) 
# np.save(os.path.join("./dress02/agglo_res/","agglo_adj"),agglo_adj) 
# np.save(os.path.join("./dress02/kmeans_res/","kmeans_adj"),kmeans_adj) 
# np.save(os.path.join("./dress02/affinity_res/","affinity_adj"),affinity_adj) 