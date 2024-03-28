# Total: 本模块用于结果数据可视化

#region
# from ursina import *
from src.obj_parser import Mesh_obj
import numpy as np
from tqdm import tqdm
import copy

from torch_geometric.utils import get_laplacian,to_dense_adj
import torch
from torch.utils.data import Dataset,DataLoader
from torch_geometric.transforms import FaceToEdge
from torch_geometric.data import Data
import os
import torch_geometric
from scipy.spatial.transform import Rotation

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import cv2
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#endregion

# region 顶点坐标随动画帧变动可视化:500帧动作下,12273个顶点的三个坐标分量变化折线图,每张图表示一个顶点。
x1 = np.load("./VirtualBoneDataset/dress02/HFdata/0.npz",allow_pickle=True)
x2 = np.load("./VirtualBoneDataset/dress02/HFdata/1.npz",allow_pickle=True)

# 绘制曲线图并保存到本地
path = "./photo/"
if not os.path.exists(path):
    os.makedirs(path)
for i in range(12273)[:]:
    plt.figure()
    plt.ylim(-1.0, 1.5)
    plt.plot(x1["sim_res"][:,i,0],label="X")
    plt.plot(x1["sim_res"][:,i,1],label="Y")
    plt.plot(x1["sim_res"][:,i,2],label="Z")
    plt.legend()
    name = "point_index = " + str(i)
    plt.title(name)
    figname = path+'pos_motion0_'+str(i)+'.png'
    plt.savefig(figname,bbox_inches='tight')   

    plt.clf()
    plt.close()

# 加载本地png生成动画文件
path = r"./photo/pos_motion0_"
fps = 24  
fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWriter = cv2.VideoWriter('TestVideo.mp4',fourcc,fps,(559,433),True)
for i in range(12273):
    frame = cv2.imread(path+str(i)+'.png')
    cv2.imshow('frame',frame)
    videoWriter.write(frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
            break
videoWriter.release()
cv2.destroyAllWindows()
# endregion

# 数据加载
#region 
pred_obj_path = "./out/"
dataset_path = "./VirtualBoneDataset/dress02/HFdata/"
picture_path = "./out/"
# anima_path = "./"
obj_template_path = "assets/dress02/garment.obj"
garment_template = Mesh_obj(obj_template_path)
out_path = "./True_obj_anima0"
state_path =  "assets/dress02/state.npz"
#endregion


# """真实数据与高频输出对齐处理,得到HF网络输出的标签用于计算loss"""
#region
#注: 使用自定义数据集必须有
#   1.K组动作序列-[帧数,poses:{52,3},trans:{3}];
#   2.布料原始模型;
#   3.布料仿真结果K组:[帧数,顶点数,3];
#   4.K组动作序列的pose均值方差,trans均值方差,以及对应布料仿真数据的均值方差;
#   5.SSDR蒙皮分解得到虚拟骨骼的pose和trans均值方差;
#   6.预训练LF后得到的低频输出"K_train"组布料仿真数据的均值和方差
#   7. ...
#state = np.load(state_path)
#for i in os.listdir(dataset_path)[:]:
#    if "npz" in i:
#        tmp = np.load(os.path.join(dataset_path,i),allow_pickle=True)
#        if (tmp["pose"].shape != (500,52,3))or(tmp["trans"].shape != (500,3))or(tmp["sim_res"].shape!=(500,12273,3)) :
#            continue
#        pose_arr = (tmp["pose"] - state["pose_mean"]) / state["pose_std"]
#        trans_arr = (tmp["trans"] - tmp["trans"][0] - state["trans_mean"]) / state["trans_std"]
#        result = np.zeros_like(tmp["sim_res"])
#        for frame in range(500)[:]:
#            pose = pose_arr[frame] * state["pose_std"] + state["pose_mean"]
#            trans = trans_arr[frame] * state["trans_std"] + state["trans_mean"]
#            trans_off = np.array([0,-2.1519510746002397,90.4766845703125]) / 100.0
#            trans += trans_off
#            final_res_ori = (tmp["sim_res"][frame] - trans).transpose() # 转置
#            t = (Rotation.from_rotvec(pose[0]).as_matrix()).transpose() # 旋转矩阵转置即为逆矩阵
#            # final_res_ori = torch.from_numpy(np.matmul(t,final_res_ori).transpose())
#            result[frame,:,:] = np.matmul(t,final_res_ori).transpose()
#       
#            #out_obj = copy.deepcopy(garment_template) 
#            #out_obj.v = result[frame,:,:] 
#            #out_obj.write(os.path.join(out_path, "{}.obj".format("Testing")))
#        np.savez(r"D:\ProgramTotal\ProgramDemo\VS2019\VirtualBone0302\VirtualBone0302\VirtualBoneDataset\dress02\HF_res/"+i,final_ground=result)
#endregion


#Laplacian平滑计算
#region 
def Smoothing_pos(edge_index,pos_x,normalization='sym'):
    """网格的Laplacian平滑"""
    # pos_x:[batchsize,numvert,3]
    # 1.计算modified laplacian平滑矩阵L
    indexs,weights = get_laplacian(edge_index,normalization=normalization)
    # 2.根据L对原始网格坐标进行平滑
    dpos = torch.zeros_like(pos_x)
    res = torch.mul(weights.reshape(1,-1,1),pos_x[:,indexs[0]])

    for i in range(pos_x.shape[1]):
        tmpp = res[:,indexs[1]==(torch.zeros_like(indexs[1])+i)]
        dpos[:,i] = torch.sum(tmpp,dim=1)
    return dpos
#endregion

#为了解决速度问题:需要重新生成一个低频网络训练集：LR_i.npz文件["pose","trans","dpos","ddpos"]
#region 
#garment_template = Mesh_obj("assets/dress02/garment.obj")
#data = FaceToEdge()(Data(num_nodes=garment_template.v.shape[0],face=torch.from_numpy(garment_template.f.astype(int).transpose() - 1).long()))
#edge_index = data.edge_index
#for i in os.listdir("./VirtualBoneDataset/dress02/HFdata/")[:]:
#    if "npz" in i:
#        tmp = np.load(os.path.join("./VirtualBoneDataset/dress02/HFdata/",i),allow_pickle=True)
#        name = "LF_"+ i
#        dpos = []
#        ddpos = []
#        if (tmp["pose"].shape != (500,52,3))or(tmp["trans"].shape != (500,3))or(tmp["sim_res"].shape!=(500,12273,3)) :
#            continue
#        for j in range(tmp["sim_res"].shape[0]):
#            dpos.append(Smoothing_pos(edge_index,torch.from_numpy(tmp["sim_res"][j]).reshape(1,12273,3)).numpy())
#            ddpos.append(Smoothing_pos(edge_index,torch.from_numpy(dpos[j])).numpy())
#        np.savez(os.path.join("./VirtualBoneDataset/dress02/LFdata/",name),pose=tmp["pose"],trans=tmp["trans"],dpos=np.array(dpos),ddpos=np.array(ddpos))
#endregion
