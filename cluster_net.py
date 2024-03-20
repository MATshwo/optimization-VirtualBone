"""
1. 尝试搭建簇分离的神经网络模型: 拆分大规模布料顶点为多个簇,分开并行进行训练
2. 初步尝试 == 基于virtualbone的预训练低频模型,对高频部分分开预测训练 == 因为LF阶段主要是对motion从人体转移到布料虚拟骨骼,然后使用LBS重建(不需要训练)
并未涉及细节的顶点坐标预测,而高频部分的图卷积网络性能受顶点规模的影响很大,所以考虑在高频采取簇分割思想展开训练。 -- 其中簇分割的理论和方法都源于概率论方法
 -- 将顶点坐标视作服从高维随机分布,不同顶点之间的差异性通过度量二者之间分布的差异性来衡量 == KL/WAassert等
"""

# 尝试多线程
from multiprocessing import Process,Queue
import torch.multiprocessing as mp
import copy
import os
import random
from tqdm import tqdm
import numpy as np
import scipy
from scipy.spatial.transform import Rotation
import json
import gc
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.transforms import FaceToEdge

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from src.obj_parser import Mesh_obj
from src.SSDRLBS import SSDRLBS
from src.models import *
from Mytools import *
from torch_geometric.utils import get_laplacian,to_dense_adj,dense_to_sparse

os.environ ["CUDA_VISIBLE_DEVICES"] = "2,1,0"

class AverageMeter():
    """Computes and stores the average and current value"""

class AverageMeter():
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def Training(config_path,out_path,cluster_id=0,device="cpu",mode="LF"):
    
    
    torch.cuda.set_device(device)
    ## 1. 配置参数加载
    with open(config_path, "r") as f:
        config = json.load(f)

    gru_dim = config["gru_dim"]
    gru_out_dim = config["gru_out_dim"]
    joint_list = config["joint_list"]
    ssdrlbs_bone_num = config["ssdrlbs_bone_num"]
    ssdrlbs_root_dir = config["ssdrlbs_root_dir"]
    state_path = config["state_path"]
    obj_template_path = config["obj_template_path"]
    cluster_affinity_path = config["affinity_message_path"]
    cluster_adj_path = config["affinity_adj_path"]


    garment_template = Mesh_obj(obj_template_path)
    joint_num = len(joint_list) #20
    state = np.load(state_path)
    cluster_affinity = np.load(cluster_affinity_path,allow_pickle=True)
    cluster_adj = torch.from_numpy(np.load(cluster_adj_path))
    
    ## 2. 读取state_npy获取"标准化"参数

    # cloth_pose/trans.shape:[80,3]对应虚拟骨骼的旋转和平移,作为ground_truth
    cloth_pose_mean = torch.from_numpy(state["cloth_pose_mean"]).to(device)
    cloth_pose_std = torch.from_numpy(state["cloth_pose_std"]).to(device)
    cloth_trans_mean = torch.from_numpy(state["cloth_trans_mean"]).to(device)
    cloth_trans_std = torch.from_numpy(state["cloth_trans_std"]).to(device)

    #[numvert,3],ssdr处理后的布料网格坐标,同样ground_truth
    ssdr_res_mean = torch.from_numpy(state["ssdr_res_mean"]).to(device)
    ssdr_res_std = torch.from_numpy(state["ssdr_res_std"]).to(device)
    #[numvert,3]根据仿真得到的布料动画坐标,作为ground_truth
    vert_std = torch.from_numpy(state["sim_res_std"]).to(device)
    vert_mean = torch.from_numpy(state["sim_res_mean"]).to(device)

    ssdrlbs_net_path = config["ssdrlbs_net_path1"]
    detail_net_path = config["detail_net_path"]
    detail_net_path_save = config["detail_net_path_save"]
    
    affinity_max = cluster_affinity["dims"][0]
    print(affinity_max)

    affinity_id_list = torch.from_numpy(cluster_affinity["index"]).to(device)
    affinity_order = torch.from_numpy(cluster_affinity["order"]).to(device)


    # 3. 加载cluster顶点数据的邻接信息

    # 4. 初始化LR模型
    ssdr_model = GRU_Model((joint_num + 1) * 3, gru_dim, [ssdrlbs_bone_num * 6]).to(device)
    ssdrlbs = SSDRLBS(os.path.join(ssdrlbs_root_dir, "u.obj"),
                      os.path.join(ssdrlbs_root_dir, "skin_weights.npy"),
                      os.path.join(ssdrlbs_root_dir, "u_trans.npy"),
                      device)


    ## 6. 加载动作序列和仿真结果作为label->需要进行标准化处理
    Batchsize_lf = 8
    Batchsize_hf = 3
    LF_epoch = 100
    HF_epoch = 100

    laplace_loss = 1.0

    # Only for test:load single anim
    # anim = np.load(anim_path)
    # pose_arr = (anim["poses"] - state["pose_mean"]) / state["pose_std"]
    # trans_arr = (anim["trans"] - anim["trans"][0] - state["trans_mean"]) / state["trans_std"]
    
    if mode == 'LF':
        # 计算拉普拉斯平滑矩阵

        data = FaceToEdge()(Data(num_nodes=garment_template.v.shape[0],
                             face=torch.from_numpy(garment_template.f.astype(int).transpose() - 1).long()))

        LFdata = Mydata(config["motion_path"],state,mode="LF")
        ssdr_model.train()
        groundLF = DataLoader(dataset=LFdata,batch_size=Batchsize_lf,shuffle=True,drop_last=True,num_workers=0)
        optimizer_lf = torch.optim.Adam(ssdr_model.parameters(), lr=0.0001) # 使用SGD效果不太好
        # scheduler_lf = torch.optim.lr_scheduler.StepLR(optimizer_lf, step_size=5, gamma=0.1)

        smoother = Laplacian_Smooth()
        loss_list = []
        data.edge_index = data.edge_index.to(device)
        for epoch in tqdm(range(LF_epoch)):
            #for _,(pose_arr,trans_arr,simData) in enumerate(groundLF):
            for _,(pose_arr,trans_arr,simData,dpos) in enumerate(groundLF):
                # gc.collect()
                # torch.cuda.empty_cache()
                ssdr_hidden = None
                loss_lf = 0.0
                item_length = pose_arr.shape[1] # 500个关键帧
                # print(pose_arr.shape) #[8,500,53,3]
                #for frame in tqdm(range(item_length)):
                for frame in (range(item_length)):
                    # shape -> [batchsize,1,21*3]
                    motion_signature = np.zeros((Batchsize_lf,(len(joint_list) + 1) * 3), dtype=np.float32)
                    for j in range(len(joint_list)):
                        motion_signature[:,j * 3: j * 3 + 3] = pose_arr[:,frame, joint_list[j]]
                    motion_signature[:,len(joint_list) * 3:] = trans_arr[:,frame]

                    motion_signature = torch.from_numpy(motion_signature)
                    motion_signature = motion_signature.view((Batchsize_lf, -1)).to(device)

                    # 估计虚拟骨骼的运动数据
                    pred_rot_trans, new_ssdr_hidden = ssdr_model(motion_signature, ssdr_hidden)
                    ssdr_hidden = new_ssdr_hidden
                    #print(pred_rot_trans.shape) #[Batchsize,480]

                    pred_pose = pred_rot_trans.view((Batchsize_lf,-1, 6))[:,:, 0:3] * cloth_pose_std + \
                                cloth_pose_mean
                    pred_trans = pred_rot_trans.view((Batchsize_lf,-1, 6))[:,:, 3:6] * cloth_trans_std + \
                                    cloth_trans_mean

                    # 返回经ssdrlbs重构后的布料模型,且已经参照state_pose进行了位置还原,可直接计算与低频标签计算loss
                    ssdr_res = ssdrlbs.batch_pose(pred_trans.reshape((Batchsize_lf, 1, pred_trans.shape[1], pred_trans.shape[2])),
                                                    torch.deg2rad(pred_pose).reshape(
                                                        (Batchsize_lf, 1, pred_pose.shape[1], pred_pose.shape[2])))
                    # print(ssdr_res.shape) #[Batchsize,1,12273,3]

                    
                    #dpred_pos = (smoother.laplacian_smooth(ssdr_res[:,0,:,:].to("cpu"),data.edge_index.to("cpu")))[9]  # 平滑10次的结果
                    is_mse = True
                    if (is_mse):
                        dpred_pos = smoother.laplacian_smooth(ssdr_res[:,0,:,:],data.edge_index,is_train=True)  # 平滑10次的结果
                        tmploss= 0.0
                        tmploss = laplace_loss*Loss_func(dpos[:,frame,:,:].to(device),dpred_pos.to(device),mode="L2")
                        loss_lf = loss_lf + Loss_func(simData[:,frame,:,:].to(device),ssdr_res[:,0,:,:].to(device),mode="L2") + tmploss.to(device)
                    else:
                        # 这里好像反了,应该是真实数据指导预测数据,KL散度左右顺序有关系,不过应该影响不大
                        # 可以测试下KL达到最小时对应的MSE距离
                        loss_lf = loss_lf + KL_distance(ssdr_res[:,0,:,:].to(device),simData[:,frame,:,:].to(device)) 

                # 一个batch计算一次loss反向传播
                loss_lf = loss_lf/(item_length)
                loss_list.append(loss_lf.item())
                
                optimizer_lf.zero_grad()
                loss_lf.backward()
                optimizer_lf.step()
            # scheduler_lf.step()
            if (epoch % 10 == 0):
                print("Epoch:{}/{} --- Loss_LF :{}".format(epoch,LF_epoch,loss_lf.item()))
            torch.save(ssdr_model.state_dict(),ssdrlbs_net_path)            
            np.save('./assets/dress02/KL2_LF0409',np.array(loss_list))
             
    elif (mode == "HF"):

        ssdr_model.load_state_dict(torch.load(ssdrlbs_net_path))
        ssdr_model.eval()
        cluster_edge_index = dense_to_sparse(cluster_adj[cluster_id])[0].to(device)
        # affinity_max = 2403
        detail_model = GRU_GNN_Model(6 * ssdrlbs_bone_num, gru_dim, [gru_out_dim * affinity_max],
                                 gru_out_dim, [3,8,16], cluster_edge_index)
        
        
        
        detail_model.to(device)
        # lr=0.000003 -> 0.03
        optimizer_hf = torch.optim.Adam(detail_model.parameters(), lr=0.01)
        # 学习率优化方案
        #scheduler_hf = torch.optim.lr_scheduler.StepLR(optimizer_hf, step_size=5, gamma=0.1)

        HFdata = cluster_data(state=state,id_list=affinity_id_list,length=affinity_max,path=config["motion_path"],cluster_id=cluster_id)
        detail_model.train()
        groundHF = DataLoader(dataset=HFdata,batch_size=Batchsize_hf,shuffle=True,drop_last=True,num_workers=0)
        loss_list = []

        for epoch in tqdm(range(HF_epoch)):
            for _,(pose_arr,trans_arr,cluster_res) in enumerate(groundHF):

                ssdr_hidden = None
                detail_hidden = None
                loss_hf = 0.0
                
                item_length = pose_arr.shape[1] 
                for frame in range(item_length):

                    with torch.no_grad():
                        motion_signature = np.zeros((Batchsize_hf,(len(joint_list) + 1) * 3), dtype=np.float32)
                        for j in range(len(joint_list)):
                            motion_signature[:,j * 3: j * 3 + 3] = pose_arr[:,frame, joint_list[j]]
                        motion_signature[:,len(joint_list) * 3:] = trans_arr[:,frame]

                        motion_signature = torch.from_numpy(motion_signature)
                        motion_signature = motion_signature.view((Batchsize_hf, -1)).to(device)

                        pred_rot_trans, new_ssdr_hidden = ssdr_model(motion_signature, ssdr_hidden)
                        ssdr_hidden = new_ssdr_hidden

                        pred_pose = pred_rot_trans.view((Batchsize_hf,-1, 6))[:,:, 0:3] * cloth_pose_std + \
                                    cloth_pose_mean
                        pred_trans = pred_rot_trans.view((Batchsize_hf,-1, 6))[:,:, 3:6] * cloth_trans_std + \
                                        cloth_trans_mean

                        ssdr_res = ssdrlbs.batch_pose(pred_trans.reshape((Batchsize_hf, 1, pred_trans.shape[1], pred_trans.shape[2])),
                                                        torch.deg2rad(pred_pose).reshape(
                                                            (Batchsize_hf, 1, pred_pose.shape[1], pred_pose.shape[2])))
                    
                        #ssdr_res.detach() 
                        #pred_rot_trans.detach()
                    # print(ssdr_res.shape) # [batch,1,12273,3]
                    ssdr_res = ssdr_res.reshape(Batchsize_hf,-1,3)
                    # 输入需要改变: ssdr_res先重排然后取当前类的成员
                    ssdr_cluster = torch.zeros(Batchsize_hf,affinity_max,3).to(device)
                    ssdr_cluster[:,:affinity_id_list[cluster_id+1]-affinity_id_list[cluster_id],:] = ssdr_res[:,affinity_order,:][:,affinity_id_list[cluster_id]:affinity_id_list[cluster_id+1],:]
                    
                    detail_res, new_detail_hidden = detail_model(pred_rot_trans, ssdr_cluster, detail_hidden)
                    detail_hidden = new_detail_hidden

                    # 预测的结果
                    #final_res = ssdr_res.detach().cpu().numpy().reshape((Batchsize_hf,-1, 3)) + \
                    #            (detail_res.detach().cpu().numpy().reshape((Batchsize_hf,-1, 3)) * ssdr_res_std + ssdr_res_mean)
                    
                    # 这俩个也需要重排,先不放入GPU,节省资源
                    cluster_std = torch.zeros(affinity_max,3).to(device)
                    cluster_mean = torch.zeros(affinity_max,3).to(device)
                    cluster_std[:affinity_id_list[cluster_id+1]-affinity_id_list[cluster_id],:] = ssdr_res_std[affinity_order,:][affinity_id_list[cluster_id]:affinity_id_list[cluster_id+1],:]
                    cluster_mean[:affinity_id_list[cluster_id+1]-affinity_id_list[cluster_id],:] = ssdr_res_mean[affinity_order,:][affinity_id_list[cluster_id]:affinity_id_list[cluster_id+1],:]


                    # 长度对齐,都是[batch,max,3]
                    final_res = ssdr_cluster.reshape((Batchsize_hf,-1, 3)) + (detail_res.reshape((Batchsize_hf,-1, 3)) * cluster_std + cluster_mean)
                    # 这里计算的是一个batch的loss
                    loss_hf = loss_hf + Loss_func(cluster_res[:,frame,:,:].to(device),final_res,mode="L2")

                    #pose = pose_arr[:,frame].to(device) * pose_std + pose_mean 
                    #trans = (trans_arr[:,frame].to(device) * trans_std + trans_mean).reshape(Batchsize_hf,1,3)
                    
                    #trans_off = torch.from_numpy(np.array([0,
                    #                      -2.1519510746002397,
                    #                      90.4766845703125]) / 100.0).to(device)
                    #trans += trans_off

                    #final_res = torch.matmul(torch.from_numpy(Rotation.from_rotvec(pose[:,0].cpu()).as_matrix().astype(np.float32)).to(device),
                    #                      final_res.transpose(2,1)).transpose(2,1)
                    #final_res += trans
                    
                    #out_obj = copy.deepcopy(garment_template)
                    #out_obj.v = final_res
                    #out_obj.write(os.path.join(out_path, "{}.obj".format(frame)))
                loss_list.append(loss_hf.item())
                optimizer_hf.zero_grad()
                loss_hf.backward()
                optimizer_hf.step()
                # del loss_hf
            #scheduler_hf.step()
            
                print("Epoch:{}/{} --- Loss_HF :{}".format(epoch,HF_epoch,loss_hf.item()))
            torch.save(detail_model.state_dict(),os.path.join(detail_net_path_save,"spectral/Cluster_cos_{}_0413.pth.tar".format(cluster_id)))
            np.save('./assets/dress02/checkpoints/cosine_distance/spectral/cluster_cos_{}_5_0413'.format(cluster_id),np.array(loss_list))
            # np.save('./assets/dress02/checkpoints/cosine_distance/spectral/cluster_cos_{}_0413'.format(cluster_id),np.array(loss_list))

    return None


if __name__ == "__main__":

    config_path = "assets/dress02/config.json"
    # anim_path = "anim/anim2.npz"
    out_path = "cluster_out"
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    #  watch -n 10 nvidia-smi : 查看GPU内存使用
    device = "cuda:2"
    
    #Training(config_path,out_path,device,mode="LF")
    cluster_num = 9
    for i in range(9)[8:]:

        print("the {}-th cluster is training...".format(i))
        Training(config_path=config_path,out_path=out_path,cluster_id = i, device=device,mode="HF")

