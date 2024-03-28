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
import matplotlib.pyplot as plt


os.environ ["CUDA_VISIBLE_DEVICES"] = "1,2,0"


def Training(config_path,out_path, device="cpu",mode="LF"):
    
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
    joint_num = len(joint_list) 
    state = np.load(state_path)
    
    cloth_pose_mean = torch.from_numpy(state["cloth_pose_mean"]).to(device)
    cloth_pose_std = torch.from_numpy(state["cloth_pose_std"]).to(device)
    cloth_trans_mean = torch.from_numpy(state["cloth_trans_mean"]).to(device)
    cloth_trans_std = torch.from_numpy(state["cloth_trans_std"]).to(device)

    ssdr_res_mean = torch.from_numpy(state["ssdr_res_mean"]).to(device)
    ssdr_res_std = torch.from_numpy(state["ssdr_res_std"]).to(device)

    vert_std = torch.from_numpy(state["sim_res_std"]).to(device)
    vert_mean = torch.from_numpy(state["sim_res_mean"]).to(device)

    ssdrlbs_net_path = config["ssdrlbs_net_path1"]
    detail_net_path = config["detail_net_path"]
    detail_net_path_save = config["detail_net_path_save"]
    
    data = FaceToEdge()(Data(num_nodes=garment_template.v.shape[0],
                             face=torch.from_numpy(garment_template.f.astype(int).transpose() - 1).long()))
    
    ssdr_model = GRU_Model((joint_num + 1) * 3, gru_dim, [ssdrlbs_bone_num * 6]).to(device)

    ssdrlbs = SSDRLBS(os.path.join(ssdrlbs_root_dir, "u.obj"),
                      os.path.join(ssdrlbs_root_dir, "skin_weights.npy"),
                      os.path.join(ssdrlbs_root_dir, "u_trans.npy"),
                      device)

    init_motion_path = "VirtualBoneDataset/dress02/HF_res/0.npz"
    init_motion = np.load(init_motion_path,allow_pickle=True)["final_ground"] #[500,12273,3]-[12273,500,3]
    numvers = init_motion.shape[1]
    init_motion = init_motion.transpose(1,0,2).reshape(numvers,-1) 

    Batchsize_lf = 8
    Batchsize_hf = 5
    LF_epoch = 50
    HF_epoch = 80
    laplace_loss = 1.0

    if mode == 'LF':
        LFdata = Mydata(config["motion_path"],state,mode="LF")
        ssdr_model.train()
        groundLF = DataLoader(dataset=LFdata,batch_size=Batchsize_lf,shuffle=True,drop_last=True,num_workers=0)
        optimizer_lf = torch.optim.Adam(ssdr_model.parameters(), lr=0.0001) # 使用SGD效果不太好
        # scheduler_lf = torch.optim.lr_scheduler.StepLR(optimizer_lf, step_size=5, gamma=0.1)

        smoother = Laplacian_Smooth()
        loss_list = []
        data.edge_index = data.edge_index.to(device)
        for epoch in tqdm(range(LF_epoch)):

            for _,(pose_arr,trans_arr,simData,dpos) in enumerate(groundLF):

                ssdr_hidden = None
                loss_lf = 0.0
                item_length = pose_arr.shape[1] # 500个关键帧
                for frame in (range(item_length)):
                    # shape -> [batchsize,1,21*3]
                    motion_signature = np.zeros((Batchsize_lf,(len(joint_list) + 1) * 3), dtype=np.float32)
                    for j in range(len(joint_list)):
                        motion_signature[:,j * 3: j * 3 + 3] = pose_arr[:,frame, joint_list[j]]
                    motion_signature[:,len(joint_list) * 3:] = trans_arr[:,frame]

                    motion_signature = torch.from_numpy(motion_signature)
                    motion_signature = motion_signature.view((Batchsize_lf, -1)).to(device)

                    pred_rot_trans, new_ssdr_hidden = ssdr_model(motion_signature, ssdr_hidden)
                    ssdr_hidden = new_ssdr_hidden


                    pred_pose = pred_rot_trans.view((Batchsize_lf,-1, 6))[:,:, 0:3] * cloth_pose_std + \
                                cloth_pose_mean
                    pred_trans = pred_rot_trans.view((Batchsize_lf,-1, 6))[:,:, 3:6] * cloth_trans_std + \
                                    cloth_trans_mean

                    ssdr_res = ssdrlbs.batch_pose(pred_trans.reshape((Batchsize_lf, 1, pred_trans.shape[1], pred_trans.shape[2])),
                                                    torch.deg2rad(pred_pose).reshape(
                                                        (Batchsize_lf, 1, pred_pose.shape[1], pred_pose.shape[2])))

                    is_mse = True
                    if (is_mse):
                        dpred_pos = smoother.laplacian_smooth(ssdr_res[:,0,:,:],data.edge_index,is_train=True)  
                        tmploss= 0.0
                        tmploss = laplace_loss*Loss_func(dpos[:,frame,:,:].to(device),dpred_pos.to(device),mode="L2")
                        loss_lf = loss_lf + Loss_func(simData[:,frame,:,:].to(device),ssdr_res[:,0,:,:].to(device),mode="L2") + tmploss.to(device)
                    else:
                        loss_lf = loss_lf + KL_distance(ssdr_res[:,0,:,:].to(device),simData[:,frame,:,:].to(device)) 

                loss_lf = loss_lf/(item_length)
                loss_list.append(loss_lf.item())
                print("Epoch:{}/{} --- Loss_LF :{}".format(epoch,LF_epoch,loss_lf.item()))
                optimizer_lf.zero_grad()
                loss_lf.backward()
                optimizer_lf.step()

            torch.save(ssdr_model.state_dict(),ssdrlbs_net_path)            
            np.save('./assets/dress02/2024/0313',np.array(loss_list))

    elif (mode == "HF"):


        vertextools = reduction_tool_plus(init_motion,method="PCA")
        dims = vertextools.init_process()
        ssdr_model.load_state_dict(torch.load(ssdrlbs_net_path,map_location="cuda:1"))
        ssdr_model.eval()

        edge_index_reduce = get_full_edge_index(dims,mode="full").to(device) 
        detail_model = MyGRU_GCN_Model_motion(6 * ssdrlbs_bone_num, gru_dim, [gru_out_dim * dims],
                                 gru_out_dim, [3,8,16], edge_index_reduce,p=0)
        

        detail_model.to(device)
        optimizer_hf = torch.optim.Adam(detail_model.parameters(), lr=0.01)
    
        HFdata = Mydata(config["motion_path"],state,mode="HF")
        detail_model.train()
        groundHF = DataLoader(dataset=HFdata,batch_size=Batchsize_hf,shuffle=True,drop_last=True,num_workers=0)
        loss_list = []
        reduce_list = []
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
                    
                    ssdr_reduce = vertextools.dim_reduction(ssdr_res) 

                    detail_reduce, new_detail_hidden,new_motion_hiddden = detail_model(pred_rot_trans,motion_signature,ssdr_reduce,detail_hidden,motion_hidden)
                    detail_hidden = new_detail_hidden
                    motion_hidden = new_motion_hiddden
                  

                    detail_res = vertextools.dim_recover(detail_reduce)
                    final_res = ssdr_res.reshape((Batchsize_hf,-1, 3)) + (detail_res.reshape((Batchsize_hf,-1, 3)) * ssdr_res_std + ssdr_res_mean)

                    reduce_truth = vertextools.dim_reduction(simData[:,frame,:,:].to(device))
                    reduce_loss = reduce_loss + Loss_func(reduce_truth.reshape(Batchsize_hf,-1),detail_reduce,mode="L2")

                    loss_hf = loss_hf + Loss_func(simData[:,frame,:,:].to(device),final_res,mode="L2")
                    
                    #out_obj = copy.deepcopy(garment_template)
                    #out_obj.v = final_res
                    #out_obj.write(os.path.join(out_path, "{}.obj".format(frame)))
                
                reduce_list.append(reduce_loss.item()/item_length)
                loss_total = reduce_loss + loss_hf
                print("Epoch:{}/{}_Loss_HF:{},reduce_loss:{}".format(epoch,HF_epoch,(loss_hf/item_length).item(),(reduce_loss/item_length).item()))
                loss_list.append(loss_hf.item()/item_length)

                optimizer_hf.zero_grad()
                loss_total.backward()
                optimizer_hf.step()

            plot_loss(loss_ori=loss_list,loss_reduce=reduce_list,lens=len(loss_list))
            torch.save(detail_model.state_dict(),os.path.join(detail_net_path_save,"PCA0313.pth.tar"))
            np.save('./assets/dress02/checkpoints/2024/PCA0313',np.array(loss_list))
    return None

def plot_loss(loss_total=None,loss_ori=None,loss_reduce=None,loss_laplacian=None,lens=0):

    train_num = lens
    f = plt.figure(figsize=(12,8))

    if loss_total != None:
        plt.plot(range(train_num),loss_total,label="Loss_total")
    if loss_ori != None:
        plt.plot(range(train_num),loss_ori,label="Loss_in_originSpace")
    if loss_reduce != None:
        plt.plot(range(train_num),loss_reduce,label="Loss_in_reductionSpace")
    if loss_laplacian != None:
        plt.plot(range(train_num),loss_laplacian,label="Loss_laplacian")
    plt.legend()
    plt.title("RMSE_Loss after {} training...".format(train_num))
    f.savefig(os.path.join("assets/dress02/checkpoints/2024",'pca0313.png'),dpi=300)
    plt.close()

if __name__ == "__main__":
    config_path = "assets/dress02/config.json"
    # anim_path = "anim/anim2.npz"
    out_path = "out"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    device = "cuda:2"
    Training(config_path,out_path,device,mode="HF")
