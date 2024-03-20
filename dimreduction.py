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

from src.obj_parser import Mesh_obj
from src.SSDRLBS import SSDRLBS
from src.models import *
from Mytools import *
from torch_geometric.utils import get_laplacian,to_dense_adj,dense_to_sparse

def get_full_edge_index(numverts=1,mode="empty"):
    # 根据顶点数量创建全孤立&全连通的邻接矩阵  -- 返回稀疏格式[2,edge_num]
    from torch_geometric.utils import dense_to_sparse    
    if mode == "empty":
        adj = torch.zeros((numverts,numverts))  # 使用全连接导致最终输出无法区分不同维度的差别，所以由全连通改为全不连通
    elif mode == "full":
        # 全连通邻接矩阵
        adj = torch.ones((numverts,numverts)) - torch.eye(numverts)
    return dense_to_sparse(adj)[0]
# 尝试还原训练过程
os.environ ["CUDA_VISIBLE_DEVICES"] = "1,2,0"
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

def Training(config_path,out_path, device="cpu",mode="LF"):
    
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
    
    garment_template = Mesh_obj(obj_template_path)
    joint_num = len(joint_list) #20
    state = np.load(state_path)
    
    ## 2. 读取state_npy获取"标准化"参数
    # state应该是训练集的均值和方差
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
    
    ## 3. 加载布料初始模型,并获取邻接信息和Laplacian矩阵
    data = FaceToEdge()(Data(num_nodes=garment_template.v.shape[0],
                             face=torch.from_numpy(garment_template.f.astype(int).transpose() - 1).long()))
    
    ## 4. 初始化LR模型和优化器
    ssdr_model = GRU_Model((joint_num + 1) * 3, gru_dim, [ssdrlbs_bone_num * 6]).to(device)

    ssdrlbs = SSDRLBS(os.path.join(ssdrlbs_root_dir, "u.obj"),
                      os.path.join(ssdrlbs_root_dir, "skin_weights.npy"),
                      os.path.join(ssdrlbs_root_dir, "u_trans.npy"),
                      device)

    ## 6. 加载动作序列和仿真结果作为label->需要进行标准化处理
    Batchsize_lf = 8
    Batchsize_hf = 5
    LF_epoch = 50
    HF_epoch = 100
    laplace_loss = 1.0

    if mode == 'LF':
        # 计算拉普拉斯平滑矩阵

        LFdata = Mydata(config["motion_path"],state,mode="LF")
        ssdr_model.train()
        groundLF = DataLoader(dataset=LFdata,batch_size=Batchsize_lf,shuffle=True,drop_last=True,num_workers=0)
        optimizer_lf = torch.optim.Adam(ssdr_model.parameters(), lr=0.0001) # 使用SGD效果不太好
        # scheduler_lf = torch.optim.lr_scheduler.StepLR(optimizer_lf, step_size=5, gamma=0.1)

        # 初始化平滑器
        
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
                # 针对每个动作序列的操作
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

                    # 因为喂入的数据在上文进行了标准化,对应预测结果也是标准化向量,需要还原
                    pred_pose = pred_rot_trans.view((Batchsize_lf,-1, 6))[:,:, 0:3] * cloth_pose_std + \
                                cloth_pose_mean
                    pred_trans = pred_rot_trans.view((Batchsize_lf,-1, 6))[:,:, 3:6] * cloth_trans_std + \
                                    cloth_trans_mean

                    # 返回经ssdrlbs重构后的布料模型,且已经参照state_pose进行了位置还原,可直接计算与低频标签计算loss
                    ssdr_res = ssdrlbs.batch_pose(pred_trans.reshape((Batchsize_lf, 1, pred_trans.shape[1], pred_trans.shape[2])),
                                                    torch.deg2rad(pred_pose).reshape(
                                                        (Batchsize_lf, 1, pred_pose.shape[1], pred_pose.shape[2])))
                    # print(ssdr_res.shape) #[Batchsize,1,12273,3]

                    # 与低频数据(并非直接与label)计算loss:平滑结果在anima.py中已计算并保存至LFdata
                    # Loss也需要更新为原始损失和laplacian损失！不然还原的布料褶皱区域会炸裂！
                    
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
                print("Epoch:{}/{} --- Loss_LF :{}".format(epoch,LF_epoch,loss_lf.item()))
                optimizer_lf.zero_grad()
                loss_lf.backward()
                optimizer_lf.step()
            # scheduler_lf.step()
            torch.save(ssdr_model.state_dict(),ssdrlbs_net_path)            
            np.save('./assets/dress02/KL2_LF0409',np.array(loss_list))

    elif (mode == "HF"):
        
        
        vertextools = reduction_tool_plus(garment_template.v,method="PCA")
        vertextools.init_process()
        # vertextools = reduction_tool(garment_template.v)
        # vertextools.PCA_process()


        ssdr_model.load_state_dict(torch.load(ssdrlbs_net_path,map_location=device))
        ssdr_model.eval()
        # data.edge_index = data.edge_index.to(device)

        # 指定PCA降维结果=3, 所以直接设置为全连通
        edge_index_reduce = torch.tensor([[0,1,0,2,1,2],[1,0,2,0,2,1]]).to(device) # 还是全连通,但通过神经网络学习一个边的权重结果
        #edge_index_reduce = get_full_edge_index(3,mode="empty").to(device)  # 全不连通 -- 独立
        # 此处输出维度要修改为PCA降维后的维度
        detail_model = MyGRU_GCN_Model_motion(6 * ssdrlbs_bone_num, gru_dim, [gru_out_dim * 3],
                                 gru_out_dim, [3,8,16], edge_index_reduce,p=0)
        
        if os.path.exists(os.path.join(detail_net_path_save,"Reduction_motion_PCA_0424.pth.tar")):
           detail_model.load_state_dict(torch.load(os.path.join(detail_net_path_save,"Reduction_motion_PCA_0424.pth.tar"))) 
           print("Loading exists model...")
        
        detail_model.to(device)
        # 0.00003的学习率更新太慢 -- 初始要用大一点的
        optimizer_hf = torch.optim.Adam(detail_model.parameters(), lr=0.005)
        #scheduler_hf = torch.optim.lr_scheduler.StepLR(optimizer_hf, step_size=2, gamma=0.1)
    
        HFdata = Mydata(config["motion_path"],state,mode="HF")
        detail_model.train()
        groundHF = DataLoader(dataset=HFdata,batch_size=Batchsize_hf,shuffle=True,drop_last=True,num_workers=0)
        loss_list = []
        for epoch in tqdm(range(HF_epoch)):
            for _,(pose_arr,trans_arr,simData) in enumerate(groundHF):

                ssdr_hidden = None
                detail_hidden = None
                motion_hidden = None
                loss_hf = 0.0
                reduce_loss = 0.0
                
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
                    
                    # print(pred_rot_trans.shape) #[5,480]
                    # ssdr_res.shape = [batch,1,12273,3]
                    ssdr_reduce = vertextools.dim_reduction(ssdr_res) # [batch,1,new_dim,3] => 直接降到了2维 :) 
                    
                    detail_reduce, new_detail_hidden,new_motion_hiddden = detail_model(pred_rot_trans,motion_signature,ssdr_reduce,detail_hidden,motion_hidden)
                    detail_hidden = new_detail_hidden
                    motion_hidden = new_motion_hiddden
                  

                    detail_res = vertextools.dim_recover(detail_reduce)
                    final_res = ssdr_res.reshape((Batchsize_hf,-1, 3)) + (detail_res.reshape((Batchsize_hf,-1, 3)) * ssdr_res_std + ssdr_res_mean)

                    # 低维空间损失计算
                    reduce_truth = vertextools.dim_reduction(simData[:,frame,:,:].to(device))
                    reduce_loss = reduce_loss + Loss_func(reduce_truth.reshape(Batchsize_hf,-1),detail_reduce,mode="L2")

                    # 总损失 = 高维损失 + 低维损失
                    loss_hf = loss_hf + Loss_func(simData[:,frame,:,:].to(device),final_res,mode="L2")
                    
                    #out_obj = copy.deepcopy(garment_template)
                    #out_obj.v = final_res
                    #out_obj.write(os.path.join(out_path, "{}.obj".format(frame)))

                loss_total = reduce_loss + loss_hf
                print("Epoch:{}/{}_Loss_HF:{},reduce_loss:{}".format(epoch,HF_epoch,(loss_hf/item_length).item(),(reduce_loss/item_length).item()))
                loss_list.append(loss_total.item())

                optimizer_hf.zero_grad()
                loss_total.backward()
                optimizer_hf.step()

            # 每个epochlr重置
            #scheduler_hf.step()
            torch.save(detail_model.state_dict(),os.path.join(detail_net_path_save,"Reduction_motion_PCA_0424.pth.tar"))
            np.save('./assets/dress02/checkpoints/cosine_distance/Reduction_motion_PCA_0424',np.array(loss_list))

    return None


if __name__ == "__main__":
    config_path = "assets/dress02/config.json"
    # anim_path = "anim/anim2.npz"
    out_path = "out"
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    #  watch -n 10 nvidia-smi : 查看GPU内存使用
    device = "cuda:0"
    
    #Training(config_path,out_path,device,mode="LF")
    Training(config_path,out_path,device,mode="HF") # 比较极限,batch=3都占用了10000mb,快超了