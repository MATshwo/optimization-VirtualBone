from torch_geometric.utils import get_laplacian,to_dense_adj
import numpy as np
from src.obj_parser import Mesh_obj
import torch
from torch.utils.data import Dataset,DataLoader
from torch_geometric.transforms import FaceToEdge
from torch_geometric.data import Data
import os
from tqdm import tqdm
import torch_geometric
from Mytools import *
import copy

# 将原始网格使用laplacian平滑
HF_path = "VirtualBoneDataset/dress02/HF_res"
LF_path = "VirtualBoneDataset/dress02/LF_res"
if not os.path.exists(LF_path):
    os.makedirs(LF_path)

cloth = Mesh_obj("assets/dress02/garment.obj")
data = FaceToEdge()(Data(num_nodes=cloth.v.shape[0],face=torch.from_numpy(cloth.f.astype(int).transpose() - 1).long()))
smoother = Laplacian_Smooth()

def laplacian_compeat(pos,edge_index):

    # for i in range(n):
    #     print("For {}-th smoothing....".format(i+1))
    
    pos_list = smoother.laplacian_smooth(torch.from_numpy(pos),edge_index)
    # for i in range(len(pos_list)):
    #     out_obj = copy.deepcopy(cloth) # 深拷贝,创建新变量副本
    #     out_obj.v = pos_list[i]
    #     out_obj.write("laplacian_{}.obj".format(i))
    # 平滑20次,第10次作为LF的ground_truth,最后一次作为拉普拉斯平滑的结果用于计算Loss
    return pos_list[9],pos_list[-1]


for file in os.listdir(HF_path)[:]:

    if "npz" in file :
        # 获取初始顶点坐标
        print(file)
        init_pos = np.load(os.path.join(HF_path,file),allow_pickle=True)["final_ground"]
        lap_pos = np.zeros_like(init_pos)
        dpos = np.zeros_like(init_pos)
        for idx in range(init_pos.shape[0])[:]:
            # out_obj = copy.deepcopy(cloth) # 深拷贝,创建新变量副本
            # out_obj.v = init_pos[idx]
            # out_obj.write("laplacian_init.obj")
            lap_pos[idx],dpos[idx] = laplacian_compeat(copy.deepcopy(init_pos[idx]),data.edge_index)
        #print(lap_pos.shape,dpos.shape)
        np.savez(os.path.join(LF_path,file),lap_pos = lap_pos,dpos = dpos)
        





