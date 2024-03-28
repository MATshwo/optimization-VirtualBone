import copy
import os
import random
from tqdm import tqdm
import numpy as np
import scipy
from scipy.spatial.transform import Rotation
import json
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.transforms import FaceToEdge
from torch_geometric.utils import dropout_adj
from Mytools import *
from src.obj_parser import Mesh_obj
from src.SSDRLBS import SSDRLBS
from src.models import *
import matplotlib.pyplot as plt


def get_results(config_path, anim_path, out_path, device="cpu"):
    with open(config_path, "r") as f:
        config = json.load(f)

    gru_dim = config["gru_dim"]
    gru_out_dim = config["gru_out_dim"]
    joint_list = config["joint_list"]

    ssdrlbs_bone_num = config["ssdrlbs_bone_num"]
    ssdrlbs_root_dir = config["ssdrlbs_root_dir"]
    ssdrlbs_net_path = config["ssdrlbs_net_path1"]
    detail_net_path = config["detail_net_path1"]
    detail_net_path_save = config["detail_net_path_save"]
    state_path = config["state_path"]

    obj_template_path = config["obj_template_path"]
    garment_template = Mesh_obj(obj_template_path)

    joint_num = len(joint_list)

    ssdr_model = GRU_Model((joint_num + 1) * 3, gru_dim, [ssdrlbs_bone_num * 6])  # 为什么是*6,因为对应虚拟骨骼的旋转和平移？
    ssdr_model.load_state_dict(torch.load(ssdrlbs_net_path,map_location=torch.device(device)))
    ssdr_model = ssdr_model.to(device)
    ssdr_model.eval()


    data = FaceToEdge()(Data(num_nodes=garment_template.v.shape[0],
                             face=torch.from_numpy(garment_template.f.astype(int).transpose() - 1).long()))


    init_motion_path = "VirtualBoneDataset/dress02/HF_res/0.npz"
    init_motion = np.load(init_motion_path,allow_pickle=True)["final_ground"] #[500,12273,3]-[12273,500,3]
    numvers = init_motion.shape[1]
    init_motion = init_motion.transpose(1,0,2).reshape(numvers,-1) 
    vertextools = reduction_tool_plus(init_motion,method="KPCA")
    dims = vertextools.init_process()

    
    edge_index_reduce = get_full_edge_index(dims,mode="full").to(device) 
    detail_model = MyGRU_GCN_Model_motion(6 * ssdrlbs_bone_num, gru_dim, [gru_out_dim * dims],
                                 gru_out_dim, [3,8,16], edge_index_reduce,p=0)

    detail_model.load_state_dict(torch.load(os.path.join("assets/dress02/checkpoints/cosine_distance/","Reduction_motion_KPCA_0426.pth.tar"),map_location="cuda:0"))
    detail_model.to(device)
    detail_model.eval()

    ssdrlbs = SSDRLBS(os.path.join(ssdrlbs_root_dir, "u.obj"),
                      os.path.join(ssdrlbs_root_dir, "skin_weights.npy"),
                      os.path.join(ssdrlbs_root_dir, "u_trans.npy"),
                      device)

    state = np.load(state_path)
    cloth_pose_mean = torch.from_numpy(state["cloth_pose_mean"]).to(device)
    cloth_pose_std = torch.from_numpy(state["cloth_pose_std"]).to(device)
    cloth_trans_mean = torch.from_numpy(state["cloth_trans_mean"]).to(device)
    cloth_trans_std = torch.from_numpy(state["cloth_trans_std"]).to(device)

    ssdr_res_mean = state["ssdr_res_mean"]
    ssdr_res_std = state["ssdr_res_std"]

    vert_std = state["sim_res_std"]
    vert_mean = state["sim_res_mean"]

    anim = np.load(anim_path)
    pose_arr = (anim["pose"] - state["pose_mean"]) / state["pose_std"]
    trans_arr = (anim["trans"] - anim["trans"][0] - state["trans_mean"]) / state["trans_std"]
    item_length = pose_arr.shape[0]

    ssdr_hidden = None
    detail_hidden = None
    motion_hidden = None
    
    temp = np.load("VirtualBoneDataset/dress02/test_cloth/83.npz",allow_pickle=True)["final_ground"]

    with torch.no_grad():
        loss_list = []
        for frame in tqdm(range(item_length)[:]):

            motion_signature = np.zeros(((len(joint_list) + 1) * 3), dtype=np.float32)
            for j in range(len(joint_list)):
                motion_signature[j * 3: j * 3 + 3] = pose_arr[frame, joint_list[j]]
            motion_signature[len(joint_list) * 3:] = trans_arr[frame]

            motion_signature = torch.from_numpy(motion_signature)
            motion_signature = motion_signature.view((1, -1)).to(device)
            pred_rot_trans, new_ssdr_hidden = ssdr_model(motion_signature, ssdr_hidden)
            ssdr_hidden = new_ssdr_hidden

            pred_pose = pred_rot_trans.view((-1, 6))[:, 0:3] * cloth_pose_std + \
                        cloth_pose_mean
            pred_trans = pred_rot_trans.view((-1, 6))[:, 3:6] * cloth_trans_std + \
                         cloth_trans_mean

            ssdr_res = ssdrlbs.batch_pose(pred_trans.reshape((1, 1, pred_trans.shape[0], pred_trans.shape[1])),
                                          torch.deg2rad(pred_pose).reshape(
                                              (1, 1, pred_pose.shape[0], pred_pose.shape[1])))

            ssdr_reduce = vertextools.dim_reduction(ssdr_res) 

            detail_reduce, new_detail_hidden,new_motion_hiddden = detail_model(pred_rot_trans,motion_signature,ssdr_reduce,detail_hidden,motion_hidden)
            detail_hidden = new_detail_hidden
            motion_hidden = new_motion_hiddden

            detail_res = vertextools.dim_recover(detail_reduce)
            final_res = ssdr_res.reshape((-1, 3)).cpu().numpy() + (detail_res.reshape((-1, 3)).cpu().numpy() * ssdr_res_std + ssdr_res_mean)
            
            #loss_list.append(Loss_func(torch.from_numpy(temp[frame]),torch.from_numpy(final_res),mode="L2"))

            out_obj = copy.deepcopy(garment_template) # 深拷贝,创建新变量副本
            ##out_obj.v = ssdr_res.detach().cpu().numpy().reshape((-1, 3))
            ##out_obj.v = temp[frame,:,:]
            out_obj.v = final_res
            out_obj.write(os.path.join(out_path, "KPCA{}.obj".format(frame)))
        


if __name__ == "__main__":
    config_path = "assets/dress02/config.json"
    # anim_path = "./VirtualBoneDataset/dress02/test_motion/83.npz"
    anim_path = "./VirtualBoneDataset/dress02/HFdata/81.npz"
    out_path = "kpca"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    device = "cuda:1"

    get_results(config_path, anim_path, out_path, device)
