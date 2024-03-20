import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch_geometric.utils import get_laplacian,to_dense_adj,dense_to_sparse
import seaborn as sns
from Mytools import *

def plot_recover_matrix(matrix=None,label=0,init_motion=None):

    print(label)
    mode_list = ["PCA","NMF","FA","TSVD","dict"]
    edge_index_reduce = get_full_edge_index(numverts=18,mode="full")

    for mode in mode_list:
        if mode in label:
            vertextools = reduction_tool_plus(init_motion,method=mode)
            dims = vertextools.init_process()
            dense_adj = to_dense_adj(edge_index = edge_index_reduce,edge_attr = torch.from_numpy(matrix))[0]
            init_adj = vertextools.recover_matrix(dense_adj)
            # print(init_adj[:10][:10])
            sns.heatmap(init_adj,cmap="Blues")
            plt.title(label)
            plt.savefig("matrix_18/Frame10_{}.png".format(label[:-4]),dpi=300,bbox_inches='tight')
            plt.clf()
            plt.close()
    



if __name__ == "__main__":
    # 加载低维空间的邻接矩阵信息
    device = "CPU"
    corr_list,label_list = [],[]
    path = "./matrix_18/"

    init_motion_path = "VirtualBoneDataset/dress02/HF_res/0.npz"
    init_motion = np.load(init_motion_path,allow_pickle=True)["final_ground"] #[500,12273,3]-[12273,500,3]
    numvers = init_motion.shape[1]
    init_motion = init_motion.transpose(1,0,2).reshape(numvers,-1) 
    

    for i in os.listdir(path):
        if "npy" in i and "weights" in i :
            corr_list.append(np.load(os.path.join(path,i)))
            label_list.append(i)
            plot_recover_matrix(corr_list[-1][-1][0],label=label_list[-1],init_motion=init_motion)

        
